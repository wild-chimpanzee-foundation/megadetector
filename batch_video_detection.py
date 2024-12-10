import os
import pickle
import argparse
import subprocess
from tqdm import tqdm
from PytorchWildlife.models.detection import MegaDetectorV6
from torch.utils.data import DataLoader
from PytorchWildlife.data import datasets as pw_data


class ExtendedMegaDetector(MegaDetectorV6):
    def __init__(
        self,
        weights=None,
        device="cpu",
        pretrained=True,
        version="yolov9c",
        additional_param=None,
    ):
        super(ExtendedMegaDetector, self).__init__(
            weights=weights, device=device, pretrained=pretrained, version=version
        )
        self.additional_param = additional_param

    def batch_video_detection(
        self,
        data_path,
        output_fps=1,
        batch_size=16,
        conf_thres=0.2,
        id_strip=None,
        save_every=5,
        output_dir="results",
    ):
        video_files = [f for f in os.listdir(data_path) if f.endswith(".mp4")]
        if not video_files:
            raise ValueError("No .mp4 files found in the specified directory.")

        os.makedirs(output_dir, exist_ok=True)

        results = {}
        video_count = 0
        pickle_file_index = 1

        with tqdm(
            total=len(video_files), desc="Processing Videos", unit="video"
        ) as video_pbar:
            for video_file in video_files:
                video_path = os.path.join(data_path, video_file)

                try:
                    # Suppress ffprobe output
                    ffprobe_command = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                    process_output = subprocess.run(
                        ffprobe_command, capture_output=True, text=True
                    )
                    frame_rate = process_output.stdout.strip()
                    num, denom = map(int, frame_rate.split("/"))
                    original_fps = num / denom
                except Exception:
                    original_fps = None
                    continue

                temp_frame_dir = os.path.join(data_path, "temp_frames")
                os.makedirs(temp_frame_dir, exist_ok=True)

                video_results = []

                try:
                    temp_frame_output = os.path.join(
                        temp_frame_dir,
                        f"{os.path.splitext(video_file)[0]}_temp_frame_%06d.jpg",
                    )

                    ffmpeg_command = [
                        "ffmpeg",
                        "-i",
                        video_path,
                        "-vf",
                        f"fps={output_fps}",
                        "-q:v",
                        "2",
                        temp_frame_output,
                        "-loglevel",
                        "error",
                    ]
                    subprocess.run(ffmpeg_command, check=True)

                    extracted_frames = sorted(os.listdir(temp_frame_dir))
                    for i, frame_file in enumerate(extracted_frames):
                        frame_number = round(i * (original_fps / output_fps))
                        new_name = os.path.join(
                            temp_frame_dir,
                            f"{os.path.splitext(video_file)[0]}_frame_{frame_number:06d}.jpg",
                        )
                        os.rename(os.path.join(temp_frame_dir, frame_file), new_name)

                    self.predictor.args.batch = batch_size
                    self.predictor.args.conf = conf_thres
                    self.predictor.args.verbose = False

                    dataset = pw_data.DetectionImageFolder(
                        temp_frame_dir,
                        transform=self.transform,
                    )

                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=0,
                        drop_last=False,
                    )

                    for batch_index, (imgs, paths, sizes) in enumerate(loader):
                        det_results = self.predictor.stream_inference(paths)
                        for idx, preds in enumerate(det_results):
                            res = self.results_generation(preds, paths[idx], id_strip)
                            size = preds.orig_shape
                            normalized_coords = [
                                [x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]]
                                for x1, y1, x2, y2 in res["detections"].xyxy
                            ]
                            res["normalized_coords"] = normalized_coords
                            video_results.append(res)

                finally:
                    for temp_file in os.listdir(temp_frame_dir):
                        os.remove(os.path.join(temp_frame_dir, temp_file))
                    os.rmdir(temp_frame_dir)

                results[video_file] = video_results
                video_count += 1

                if video_count >= save_every:
                    output_file = os.path.join(
                        output_dir, f"results_part_{pickle_file_index}.pkl"
                    )
                    with open(output_file, "wb") as f:
                        pickle.dump(results, f)
                    results.clear()
                    video_count = 0
                    pickle_file_index += 1

                video_pbar.update(1)

        if results:
            output_file = os.path.join(
                output_dir, f"results_part_{pickle_file_index}.pkl"
            )
            with open(output_file, "wb") as f:
                pickle.dump(results, f)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ExtendedMegaDetector on video files."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the directory containing .mp4 files.",
    )
    parser.add_argument(
        "--weights", required=True, help="Path to the model weights file."
    )
    parser.add_argument(
        "--output_fps", type=int, default=1, help="Frames per second to extract."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference."
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.2,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save results every N videos."
    )
    parser.add_argument(
        "--output_dir", default="results", help="Directory to save output pickle files."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (e.g., 'cpu' or 'cuda').",
    )

    args = parser.parse_args()

    detector = ExtendedMegaDetector(weights=args.weights, device=args.device)

    detector.batch_video_detection(
        data_path=args.data_path,
        output_fps=args.output_fps,
        batch_size=args.batch_size,
        conf_thres=args.conf_thres,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
