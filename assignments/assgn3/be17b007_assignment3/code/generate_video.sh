ffmpeg -framerate 10 -i chn_0_%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p chn_0.mp4

ffmpeg -framerate 10 -i chn_25_%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p chn_25.mp4

ffmpeg -framerate 10 -i chn_50_%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p chn_50.mp4

ffmpeg -framerate 10 -i chn_80_%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p chn_80.mp4