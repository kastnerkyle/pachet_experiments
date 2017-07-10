file=$1
timidity --output-24bit -Ow $file
bn=${file%.mid}
ffmpeg -i $bn.wav -acodec pcm_s16le -ar 44100 $bn_1.wav
mv $bn_1.wav $bn.wav
