
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FitResult:
    left_fit: Optional[np.ndarray]
    right_fit: Optional[np.ndarray]
    left_pixels: int
    right_pixels: int
    used_sliding_windows: bool


class ADASLaneDetector:
    def __init__(
        self,
        process_width: int = 1280,
        smoothing: float = 0.2,
        expected_lane_width_m: float = 3.7,
    ) -> None:
        self.process_width = process_width
        self.smoothing = smoothing
        self.expected_lane_width_m = expected_lane_width_m

        self.left_fit: Optional[np.ndarray] = None
        self.right_fit: Optional[np.ndarray] = None
        self.prev_left_fit: Optional[np.ndarray] = None
        self.prev_right_fit: Optional[np.ndarray] = None

        self.last_confidence: float = 0.0
        self.last_lane_width_px: Optional[float] = None
        self.last_offset_m: float = 0.0
        self.last_steering: float = 0.0

        self.ym_per_pix = 30.0 / 720.0

    def resize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        scale = self.process_width / float(width)
        resized = cv2.resize(
            frame,
            (self.process_width, int(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, scale

    def threshold_lane_pixels(self, frame: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        b_channel = lab[:, :, 2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        l_channel = clahe.apply(l_channel)

        white_mask = cv2.inRange(l_channel, 180, 255)
        yellow_mask = cv2.inRange(b_channel, 155, 255)
        saturation_mask = cv2.inRange(s_channel, 60, 255)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = np.absolute(sobel_x)
        if abs_sobel_x.max() > 0:
            scaled = np.uint8(255 * abs_sobel_x / abs_sobel_x.max())
        else:
            scaled = np.zeros_like(gray)

        gradient_mask = cv2.inRange(scaled, 25, 255)

        combined = np.zeros_like(gray)
        combined[
            (white_mask > 0)
            | ((yellow_mask > 0) & (saturation_mask > 0))
            | (gradient_mask > 0)
        ] = 255

        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        return combined

    def get_roi_and_warp(
        self,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = np.float32(
            [
                [width * 0.43, height * 0.64],
                [width * 0.57, height * 0.64],
                [width * 0.88, height * 0.95],
                [width * 0.12, height * 0.95],
            ]
        )

        dst = np.float32(
            [
                [width * 0.30, 0],
                [width * 0.70, 0],
                [width * 0.70, height],
                [width * 0.30, height],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src, dst)
        inverse = cv2.getPerspectiveTransform(dst, src)
        return src, matrix, inverse

    def apply_roi(self, binary: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(binary)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        return cv2.bitwise_and(binary, mask)

    def warp_binary(
        self,
        binary: np.ndarray,
        matrix: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        return cv2.warpPerspective(binary, matrix, size, flags=cv2.INTER_LINEAR)

    def fit_polynomial_sliding_windows(self, binary_warped: np.ndarray) -> FitResult:
        height, width = binary_warped.shape
        histogram = np.sum(binary_warped[height // 2 :, :], axis=0)
        midpoint = width // 2

        leftx_base = int(np.argmax(histogram[:midpoint]))
        rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        nwindows = 9
        margin = int(width * 0.08)
        minpix = 50
        window_height = height // nwindows

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = (
            np.concatenate(left_lane_inds)
            if left_lane_inds
            else np.array([], dtype=np.int64)
        )
        right_lane_inds = (
            np.concatenate(right_lane_inds)
            if right_lane_inds
            else np.array([], dtype=np.int64)
        )

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 400 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 400 else None

        return FitResult(left_fit, right_fit, len(leftx), len(rightx), True)

    def fit_polynomial_search_around(self, binary_warped: np.ndarray) -> FitResult:
        if self.left_fit is None or self.right_fit is None:
            return self.fit_polynomial_sliding_windows(binary_warped)

        margin = int(binary_warped.shape[1] * 0.06)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            (
                nonzerox
                > (
                    self.left_fit[0] * (nonzeroy ** 2)
                    + self.left_fit[1] * nonzeroy
                    + self.left_fit[2]
                    - margin
                )
            )
            & (
                nonzerox
                < (
                    self.left_fit[0] * (nonzeroy ** 2)
                    + self.left_fit[1] * nonzeroy
                    + self.left_fit[2]
                    + margin
                )
            )
        )

        right_lane_inds = (
            (
                nonzerox
                > (
                    self.right_fit[0] * (nonzeroy ** 2)
                    + self.right_fit[1] * nonzeroy
                    + self.right_fit[2]
                    - margin
                )
            )
            & (
                nonzerox
                < (
                    self.right_fit[0] * (nonzeroy ** 2)
                    + self.right_fit[1] * nonzeroy
                    + self.right_fit[2]
                    + margin
                )
            )
        )

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 350 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 350 else None

        return FitResult(left_fit, right_fit, len(leftx), len(rightx), False)

    def estimate_lane_width_px(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        y_eval: int,
    ) -> Optional[float]:
        if left_fit is None or right_fit is None:
            return self.last_lane_width_px

        left_x = float(np.polyval(left_fit, y_eval))
        right_x = float(np.polyval(right_fit, y_eval))
        lane_width_px = right_x - left_x

        if lane_width_px <= 0:
            return self.last_lane_width_px

        self.last_lane_width_px = lane_width_px
        return lane_width_px

    def repair_missing_lane(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        lane_width_px = self.last_lane_width_px or 700.0

        if left_fit is None and right_fit is not None:
            left_fit = right_fit.copy()
            left_fit[2] -= lane_width_px
        elif right_fit is None and left_fit is not None:
            right_fit = left_fit.copy()
            right_fit[2] += lane_width_px

        return left_fit, right_fit

    def sanity_check(
        self,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        height: int,
        width: int,
    ) -> Tuple[bool, float]:
        if left_fit is None or right_fit is None:
            return False, 0.2

        y_bottom = height - 1
        y_top = int(height * 0.6)

        left_bottom = float(np.polyval(left_fit, y_bottom))
        right_bottom = float(np.polyval(right_fit, y_bottom))
        left_top = float(np.polyval(left_fit, y_top))
        right_top = float(np.polyval(right_fit, y_top))

        lane_width_bottom = right_bottom - left_bottom
        lane_width_top = right_top - left_top

        width_ok = (width * 0.22) < lane_width_bottom < (width * 0.55)
        parallel_ok = abs(lane_width_bottom - lane_width_top) < (width * 0.12)
        order_ok = left_bottom < right_bottom and left_top < right_top

        confidence = 0.0
        confidence += 0.4 if width_ok else 0.0
        confidence += 0.3 if parallel_ok else 0.0
        confidence += 0.3 if order_ok else 0.0
        return width_ok and parallel_ok and order_ok, confidence

    def smooth_fit(
        self,
        previous: Optional[np.ndarray],
        current: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if current is None:
            return previous
        if previous is None:
            return current
        return (1.0 - self.smoothing) * previous + self.smoothing * current

    def compute_curvature_and_offset(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        height: int,
        width: int,
    ) -> Tuple[float, float]:
        ploty = np.linspace(0, height - 1, height)

        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        lane_width_px = max(float(np.mean(rightx[-50:] - leftx[-50:])), 1.0)
        xm_per_pix = self.expected_lane_width_m / lane_width_px
        ym_per_pix = self.ym_per_pix

        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        y_eval_m = float(np.max(ploty) * ym_per_pix)

        def curvature_radius(fit_cr: np.ndarray) -> float:
            a, b, _ = fit_cr
            denom = max(abs(2 * a), 1e-6)
            return float(((1 + (2 * a * y_eval_m + b) ** 2) ** 1.5) / denom)

        left_curvature = curvature_radius(left_fit_cr)
        right_curvature = curvature_radius(right_fit_cr)
        curvature = (left_curvature + right_curvature) / 2.0

        lane_center_px = (leftx[-1] + rightx[-1]) / 2.0
        car_center_px = width / 2.0
        offset_m = (car_center_px - lane_center_px) * xm_per_pix

        return float(curvature), float(offset_m)

    def compute_heading_error(
        self,
        left_fit: np.ndarray,
        right_fit: np.ndarray,
        height: int,
    ) -> float:
        y_far = int(height * 0.62)
        y_near = height - 1

        left_near = float(np.polyval(left_fit, y_near))
        right_near = float(np.polyval(right_fit, y_near))
        left_far = float(np.polyval(left_fit, y_far))
        right_far = float(np.polyval(right_fit, y_far))

        center_near = (left_near + right_near) / 2.0
        center_far = (left_far + right_far) / 2.0

        dx = center_far - center_near
        dy = y_near - y_far
        return float(np.arctan2(dx, max(dy, 1)))

    def make_decision(
        self,
        offset_m: float,
        heading_error_rad: float,
        confidence: float,
    ) -> Tuple[float, str]:
        if confidence < 0.45:
            return 0.0, "hold/low-confidence"

        steering = 0.65 * offset_m + 0.35 * heading_error_rad * 5.0
        steering = float(np.clip(steering, -1.0, 1.0))

        if steering > 0.12:
            decision = "steer-left"
        elif steering < -0.12:
            decision = "steer-right"
        else:
            decision = "keep-straight"

        return steering, decision

    def draw_overlay(
        self,
        frame: np.ndarray,
        left_fit: Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        inverse: np.ndarray,
        confidence: float,
        curvature_m: Optional[float],
        offset_m: float,
        steering: float,
        decision: str,
        roi_src: np.ndarray,
    ) -> np.ndarray:
        output = frame.copy()
        height, width = frame.shape[:2]

        cv2.polylines(output, [roi_src.astype(np.int32)], True, (255, 0, 0), 2)

        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, height - 1, height)
            leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            lane_img = np.zeros_like(frame)
            left_pts = np.array([np.transpose(np.vstack([leftx, ploty]))], dtype=np.int32)
            right_pts = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))], dtype=np.int32)
            lane_pts = np.hstack((left_pts, right_pts))

            cv2.fillPoly(lane_img, lane_pts, (0, 120, 0))
            cv2.polylines(lane_img, left_pts, False, (0, 255, 255), 8)
            cv2.polylines(lane_img, np.array([np.flipud(right_pts[0])]), False, (0, 255, 0), 8)

            unwarped = cv2.warpPerspective(lane_img, inverse, (width, height))
            output = cv2.addWeighted(output, 1.0, unwarped, 0.75, 0)

        rows = [
            f"confidence: {confidence:.2f}",
            f"offset: {offset_m:+.2f} m",
            f"steering: {steering:+.2f}",
            f"decision: {decision}",
        ]
        if curvature_m is not None:
            rows.insert(1, f"curvature: {curvature_m:.0f} m")

        y = 40
        for row in rows:
            cv2.putText(
                output,
                row,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 34

        return output

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        resized, _ = self.resize_frame(frame)
        height, width = resized.shape[:2]

        binary = self.threshold_lane_pixels(resized)
        roi_src, matrix, inverse = self.get_roi_and_warp(width, height)
        binary_roi = self.apply_roi(binary, roi_src)
        binary_warped = self.warp_binary(binary_roi, matrix, (width, height))

        if self.last_confidence >= 0.45:
            fit_result = self.fit_polynomial_search_around(binary_warped)
        else:
            fit_result = self.fit_polynomial_sliding_windows(binary_warped)

        candidate_left = fit_result.left_fit
        candidate_right = fit_result.right_fit
        candidate_left, candidate_right = self.repair_missing_lane(
            candidate_left,
            candidate_right,
        )

        is_valid, geometry_conf = self.sanity_check(
            candidate_left,
            candidate_right,
            height,
            width,
        )
        pixel_conf = min(1.0, (fit_result.left_pixels + fit_result.right_pixels) / 6000.0)
        confidence = 0.55 * geometry_conf + 0.45 * pixel_conf

        if is_valid:
            self.left_fit = self.smooth_fit(self.left_fit, candidate_left)
            self.right_fit = self.smooth_fit(self.right_fit, candidate_right)
            self.prev_left_fit = self.left_fit
            self.prev_right_fit = self.right_fit
            self.estimate_lane_width_px(self.left_fit, self.right_fit, height - 1)
        else:
            self.left_fit = self.prev_left_fit
            self.right_fit = self.prev_right_fit
            confidence *= 0.6

        self.last_confidence = confidence

        curvature_m: Optional[float] = None
        offset_m = self.last_offset_m
        steering = 0.0
        decision = "hold/low-confidence"

        if self.left_fit is not None and self.right_fit is not None:
            curvature_m, offset_m = self.compute_curvature_and_offset(
                self.left_fit,
                self.right_fit,
                height,
                width,
            )
            heading_error = self.compute_heading_error(
                self.left_fit,
                self.right_fit,
                height,
            )
            steering, decision = self.make_decision(offset_m, heading_error, confidence)
            self.last_offset_m = offset_m
            self.last_steering = steering

        output = self.draw_overlay(
            resized,
            self.left_fit,
            self.right_fit,
            inverse,
            confidence,
            curvature_m,
            offset_m,
            steering,
            decision,
            roi_src,
        )

        return cv2.resize(
            output,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )


def process_video(
    input_path: str,
    output_path: Optional[str] = None,
    display: bool = True,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = ADASLaneDetector(process_width=1280, smoothing=0.20)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.process_frame(frame)

        if writer is not None:
            writer.write(result)

        if display:
            cv2.namedWindow("ADAS Lane Assist", cv2.WINDOW_NORMAL)
            cv2.imshow("ADAS Lane Assist", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = "lane-detection-input.mp4"
    output_video = "lane-detection-adas-output.mp4"
    process_video(input_video, output_path=output_video, display=True)
