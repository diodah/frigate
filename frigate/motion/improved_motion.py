import logging

import cv2
import imutils
import numpy as np
from scipy.ndimage import gaussian_filter

from frigate.comms.config_updater import ConfigSubscriber
from frigate.config import MotionConfig
from frigate.motion import MotionDetector

logger = logging.getLogger(__name__)


class ImprovedMotionDetector(MotionDetector):
    def __init__(
        self,
        frame_shape,
        config: MotionConfig,
        fps: int,
        name="improved",
        blur_radius=1,
        interpolation=cv2.INTER_NEAREST,
        contrast_frame_history=50,
    ):
        self.name = name
        self.config = config
        self.frame_shape = frame_shape
        self.resize_factor = frame_shape[0] / config.frame_height
        self.motion_frame_size = (
            config.frame_height,
            config.frame_height * frame_shape[1] // frame_shape[0],
        )
        self.avg_frame = np.zeros(self.motion_frame_size, np.float32)
        self.motion_frame_count = 0
        self.frame_counter = 0
        resized_mask = cv2.resize(
            config.mask,
            dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        self.mask = np.where(resized_mask == [0])
        self.save_images = False
        self.calibrating = True
        self.blur_radius = blur_radius
        self.interpolation = interpolation
        self.contrast_values = np.zeros((contrast_frame_history, 2), np.uint8)
        self.contrast_values[:, 1:2] = 255
        self.contrast_values_index = 0
        self.config_subscriber = ConfigSubscriber(f"config/motion/{name}")
        self.centroid_history = []
        self.magnitude_threshold = config.magnitude_threshold
        self.stability_threshold = config.stability_threshold
        self.max_history = 200

    def is_calibrating(self):
        return self.calibrating

    def detect(self, frame):
        motion_boxes = []

        self.motion_history.append(motion_boxes)

        if len(self.motion_history) > 10:
            self.motion_history.pop(0)

        if len(self.motion_history) > 1:
            prev_frame_boxes = self.motion_history[-2]
            current_frame_boxes = motion_boxes

            for prev_box, current_box in zip(prev_frame_boxes, current_frame_boxes):
                velocity, direction = self.calculate_velocity_and_direction(
                    prev_box, current_box
                )
                logger.info(
                    f"Velocity: {velocity:.2f} pixels/frame, Direction: {direction:.2f}"
                )

        # check for updated motion config
        _, updated_motion_config = self.config_subscriber.check_for_update()

        if updated_motion_config:
            self.config = updated_motion_config

        if not self.config.enabled:
            return motion_boxes

        gray = frame[0 : self.frame_shape[0], 0 : self.frame_shape[1]]

        # resize frame
        resized_frame = cv2.resize(
            gray,
            dsize=(self.motion_frame_size[1], self.motion_frame_size[0]),
            interpolation=self.interpolation,
        )

        if self.save_images:
            resized_saved = resized_frame.copy()

        # Improve contrast
        if self.config.improve_contrast:
            # TODO tracking moving average of min/max to avoid sudden contrast changes
            min_value = np.percentile(resized_frame, 4).astype(np.uint8)
            max_value = np.percentile(resized_frame, 96).astype(np.uint8)
            # skip contrast calcs if the image is a single color
            if min_value < max_value:
                # keep track of the last 50 contrast values
                self.contrast_values[self.contrast_values_index] = [
                    min_value,
                    max_value,
                ]
                self.contrast_values_index += 1
                if self.contrast_values_index == len(self.contrast_values):
                    self.contrast_values_index = 0

                avg_min, avg_max = np.mean(self.contrast_values, axis=0)

                resized_frame = np.clip(resized_frame, avg_min, avg_max)
                resized_frame = (
                    ((resized_frame - avg_min) / (avg_max - avg_min)) * 255
                ).astype(np.uint8)

        if self.save_images:
            contrasted_saved = resized_frame.copy()

        # mask frame
        # this has to come after contrast improvement
        # Setting masked pixels to zero, to match the average frame at startup
        resized_frame[self.mask] = [0]

        resized_frame = gaussian_filter(resized_frame, sigma=1, radius=self.blur_radius)

        if self.save_images:
            blurred_saved = resized_frame.copy()

        if self.save_images or self.calibrating:
            self.frame_counter += 1
        # compare to average
        frameDelta = cv2.absdiff(resized_frame, cv2.convertScaleAbs(self.avg_frame))

        # compute the threshold image for the current frame
        thresh = cv2.threshold(
            frameDelta, self.config.threshold, 255, cv2.THRESH_BINARY
        )[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh_dilated = cv2.dilate(thresh, None, iterations=1)
        contours = cv2.findContours(
            thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)

        # loop over the contours
        total_contour_area = 0
        for c in contours:
            # if the contour is big enough, count it as motion
            contour_area = cv2.contourArea(c)
            total_contour_area += contour_area
            if contour_area > self.config.contour_area:
                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append(
                    (
                        int(x * self.resize_factor),
                        int(y * self.resize_factor),
                        int((x + w) * self.resize_factor),
                        int((y + h) * self.resize_factor),
                    )
                )

        pct_motion = total_contour_area / (
            self.motion_frame_size[0] * self.motion_frame_size[1]
        )

        # once the motion is less than 5% and the number of contours is < 4, assume its calibrated
        if pct_motion < 0.05 and len(motion_boxes) <= 4:
            self.calibrating = False

        # if calibrating or the motion contours are > 80% of the image area (lightning, ir, ptz) recalibrate
        if self.calibrating or pct_motion > self.config.lightning_threshold:
            self.calibrating = True

        if self.save_images:
            thresh_dilated = cv2.cvtColor(thresh_dilated, cv2.COLOR_GRAY2BGR)
            for b in motion_boxes:
                cv2.rectangle(
                    thresh_dilated,
                    (int(b[0] / self.resize_factor), int(b[1] / self.resize_factor)),
                    (int(b[2] / self.resize_factor), int(b[3] / self.resize_factor)),
                    (255, 0, 255),
                    2,
                )
            frames = [
                cv2.cvtColor(resized_saved, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(contrasted_saved, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(blurred_saved, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
                thresh_dilated,
            ]
            cv2.imwrite(
                f"debug/frames/{self.name}-{self.frame_counter}.jpg",
                (
                    cv2.hconcat(frames)
                    if self.frame_shape[0] > self.frame_shape[1]
                    else cv2.vconcat(frames)
                ),
            )

        if len(motion_boxes) > 0:
            self.motion_frame_count += 1
            if self.motion_frame_count >= 10:
                # only average in the current frame if the difference persists for a bit
                cv2.accumulateWeighted(
                    resized_frame,
                    self.avg_frame,
                    0.2 if self.calibrating else self.config.frame_alpha,
                )
        else:
            # when no motion, just keep averaging the frames together
            cv2.accumulateWeighted(
                resized_frame,
                self.avg_frame,
                0.2 if self.calibrating else self.config.frame_alpha,
            )
            self.motion_frame_count = 0

        frame_centroids = []
        for c in contours:
            contour_area = cv2.contourArea(c)
            if (
                contour_area > self.config.contour_area
                and contour_area < total_contour_area
            ):
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    frame_centroids.append((cx, cy))

        self.centroid_history.append(frame_centroids)
        # if len(self.centroid_history) > 0:
        #     logger.info(f"Centroid history: {self.centroid_history}")
        # else:
        #     logger.info("No centroids found.")
        if len(self.centroid_history) > self.max_history:
            self.centroid_history.pop(0)

        if self.detect_suspicious_motion():
            logger.info(f"Suspicious motion detected in {self.name}")

        if self.save_images:  # Solo si se guarda la imagen o si debug lo requiere
            # Copia el frame original en color para debug
            debug_frame = cv2.cvtColor(resized_frame.copy(), cv2.COLOR_GRAY2BGR)

            # Recorre el historial de centroides
            for i in range(
                1, len(self.centroid_history)
            ):  # Empieza desde el segundo frame
                prev_centroids = self.centroid_history[i - 1]  # Centroides anteriores
                curr_centroids = self.centroid_history[i]  # Centroides actuales

                # Verificar si hay centroides tanto en el frame actual como el anterior
                if prev_centroids and curr_centroids:
                    for prev, curr in zip(prev_centroids, curr_centroids):
                        # Dibujar línea entre centroides consecutivos
                        cv2.line(debug_frame, prev, curr, (255, 0, 255), 2)

                # Dibujar los centroides actuales como puntos
                for centroid in curr_centroids:
                    cv2.circle(debug_frame, centroid, 3, (0, 255, 0), -1)

        return motion_boxes

    def detect_suspicious_motion(self):
        if len(self.centroid_history) < 2:
            return False

        directions = []
        for i in range(1, len(self.centroid_history)):
            prev_frame = self.centroid_history[i - 1]
            curr_frame = self.centroid_history[i]
            if prev_frame and curr_frame:
                for p, c in zip(prev_frame, curr_frame):
                    dx = c[0] - p[0]
                    dy = c[1] - p[1]
                    direction = np.array([dx, dy])
                    directions.append(direction)

            if directions:
                avg_direction = np.mean(directions, axis=0)
                magnitude = np.linalg.norm(avg_direction)
                stability = np.std([np.linalg.norm(d) for d in directions])
                if magnitude > 0.0 or stability > 0.0:
                    logger.info(
                        f"[{self.name}] Avg direction: {avg_direction}, Magnitude: {magnitude}, Stability: {stability}"
                    )
                    if (
                        magnitude > self.magnitude_threshold
                        and stability > self.stability_threshold
                    ):
                        logger.info(f"Suspicious motion detected in {self.name}")

            return False

    def stop(self) -> None:
        """stop the motion detector."""
        self.config_subscriber.stop()
