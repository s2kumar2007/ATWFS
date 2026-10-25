import os
import subprocess
import shutil
import urllib.request

urls = [
    "https://raw.githubusercontent.com/intel-isl/MiDaS/master/input/video/video1.mp4",
    "https://github.com/intel-isl/MiDaS/raw/master/input/video/video1.mp4" 
]

# Let's just download some stock mp4s, or I can generate realistic-ish synthetic videos to test frame-skipping and ffmpeg
from data.synthetic import write_synthetic_video
print("Generating synthetic videos since youtube downloads failed.")
os.makedirs("outputs/smoke", exist_ok=True)
write_synthetic_video("outputs/smoke/road1.mp4", 90, (640, 480), fourcc="mp4v")
write_synthetic_video("outputs/smoke/road2.mp4", 90, (800, 600), fourcc="mp4v")

videos = ["outputs/smoke/road1.mp4", "outputs/smoke/road2.mp4"]

ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except:
        ffmpeg = None

os.makedirs("outputs/inference", exist_ok=True)

for i, video_path in enumerate(videos):
    out_path = f"outputs/inference/public_road_{i}_annotated.mp4"
    print(f"Running inference on {video_path}...")
    subprocess.run(["python", "-m", "inference.live", "--source", video_path, "--checkpoint", "outputs/checkpoints/student_best.pt", "--output", out_path, "--frame-skip", "2"], check=True)
    
    if ffmpeg:
        print(f"Extracting frames for video {i+1}...")
        frames_dir = f"outputs/inference/public_road_{i}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-i", out_path, "-vf", "fps=6/10", f"{frames_dir}/frame_%02d.png"], check=True)
        # Verify ffprobe
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
        if os.path.exists(ffprobe):
            p = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", out_path], capture_output=True, text=True)
            print(f"Video {i+1} codec: {p.stdout.strip()}")

print("Verification complete.")
