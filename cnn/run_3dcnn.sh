data="/data/berisha_lab/neuwirth/data/mnf16"
masks="/data/berisha_lab/neuwirth/annotations_4_nolymphocytes"
checkpoint="/data/berisha_lab/neuwirth/code/2dcnn-keras/checkpoint-nolymphocytes"
crop=11
classes=5
samples=100000
epochs=10
batch=128

python hsi_cnn_train_keras.py --data $data --masks $masks/train --crop $crop --checkpoint $checkpoint --classes $classes --epochs $epochs --batch $batch --balance --samples $samples --validate --valdata $data --valmasks $masks/val --valsamples $samples

python hsi_cnn_classify_keras.py --data $data --masks $masks/test --checkpoint $checkpoint --crop $crop --classes $classes --metrics cnn_full