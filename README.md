# SignSpeak

A two-glove wearable that turns Ghanaian/American Sign Language fingerspelling
and a small set of two-handed word signs into spoken English.

Final year project, Department of Computer Engineering, University of Ghana.

## What it does

Put on both gloves. Spell a word letter by letter and the laptop speaks it.
Raise the left hand and sign one of ten whole-word signs and it speaks that
instead. No camera, no fixed setup, and about $135 of parts that are available
locally.

It recognises the full 26-letter alphabet plus two control gestures, and ten
two-handed word signs plus an idle class. It does not interpret facial
expression or spatial grammar, and it does not do continuous translation. Signs
are made one at a time with boundaries marked by the signer.

## Results

Letter model, 28 classes, 1,270 repetitions, 254 held out:

| model | 5-fold CV | test accuracy | macro F1 | inference |
|---|---|---|---|---|
| **LDA (lsqr + shrinkage)** | 0.9921 ± 0.0024 | **99.21%** | 0.9924 | 0.13 ms |
| Random forest | 0.9961 ± 0.0037 | 99.21% | 0.9923 | 16.29 ms |
| Logistic regression | 0.9842 ± 0.0096 | 99.61% | 0.9962 | 0.12 ms |
| SVM-RBF | 0.9833 ± 0.0086 | 98.43% | 0.9845 | 0.17 ms |

Word model, 11 classes (ten words and an idle class), 594 repetitions, 119 held out: LDA reaches 100% with
0.13 ms inference.

LDA was selected over the alternatives on cross-validation stability and on
behaviour under calibration drift, not on the test figure. Perturbing the flex
features to simulate a recalibration collapses SVM-RBF from 0.93 to 0.08 while
leaving LDA unmoved, because a uniform shift moves every class mean equally and
LDA depends on the differences between them.

A BiLSTM was trained for comparison and reaches the same
accuracy at 922 ms per window, against a 160 ms decision stride. It is six
times over budget and cannot be deployed here. An ablation on the word set
found that order-free summary statistics score higher than the order-preserving
representation, so there is no temporal structure for a recurrent model to
learn.

## Hardware

Each glove carries an ESP32-WROOM, five resistive flex sensors in 47 kΩ voltage
dividers, and an MPU-6050 over I²C. The right glove adds two copper contact
pads. Both connect to the host over USB, which supplies power and carries data.

All five flex channels sit on ADC1 pins. ADC2 shares hardware with the WiFi
radio and returns corrupted readings whenever the radio is active.

The contact pads exist because flex sensors measure how far a finger bends and
nothing else. A and T differ by thumb position, U and V by finger spread, and
neither difference is visible to a bend sensor. With the pads disabled the
letter model scored 0.90 and most of its errors were those two pairs. With them
enabled it scores 0.99.

Right glove streams 13 channels, left glove 11.

## How it works

Both gloves sample at 50 Hz. The host timestamps every frame on arrival and
pairs each right frame with the nearest left frame in either direction, which
bounds the pairing error at half a sample period. A pair further apart than
45 ms is discarded. Measured median gap is 17.6 ms with a 0.33% drop rate.

Paired frames fill a one-second window that advances every 160 ms. An arbiter
compares the left glove against a rest template captured at startup and routes
the window to the letter path when the hand is down, or the word path when it
is raised. The mode switch is measured rather than trained, so it needs no data
and cannot be confused with a real sign.

Both paths gate before classifying, on inverted rules. A letter is a held
shape, so a window with finger motion in it is rejected. A word sign is a
movement, so a window without motion is rejected. Surviving windows become five
statistics per channel, 65 features for letters and 110 for words, and go to
LDA.

Predictions are debounced before they are emitted. Letters require four
agreeing frames and then latch until the hand is released. Words require two,
followed by a refractory period. Accepted tokens collect in a buffer where
single letters run together and whole words stay separate. Signing SPEAK
flushes the buffer to sentence construction, which uses Gemini and falls back
to offline wordninja segmentation without a network, and then to pyttsx3 for
audio.

## Repository layout

```
firmware/signspeak_glove/   one sketch for both gloves, GLOVE_SIDE selects
features.py                 channel order and feature extraction
                            imported by training and by live inference
signspeak/                  host package
  dual.py                   reader threads, pairing, arbiter, gates, debounce
  glove.py                  single-glove link and the Classifier wrapper
  pipeline.py               token buffer and the runtime loop
  sentence.py               token stream to sentence
  tts.py                    speech output
  config.py                 every threshold quoted in the report
  main.py                   entry point
record_dataset.py           letter recording
record_words.py             two-glove word recording
build_features.py           raw rows to the per-repetition feature matrix
train_classical.py          letter benchmark
train_words.py              word benchmark
train_deep.py               CNN and BiLSTM, reuses the classical split
check_words.py              freeze, saturation and corrupt-read audits
sync_check.py               inter-glove frame gap
```

## Running it

```bash
pip install -r requirements.txt
cp env.example .env        # then paste your own Gemini key into .env
```

Flash `firmware/signspeak_glove/signspeak_glove.ino` to each board, setting
`GLOVE_SIDE` to `GLOVE_RIGHT` or `GLOVE_LEFT` first. The boot banner reports
the side and channel count.

Identify which port is which glove, then run:

```bash
python which_glove.py
python -m signspeak.main --port COM6 --left-port COM7 --model lda --word-model lda
```

Startup walks each glove through calibration, 150 samples flat and 150 in a
fist, then captures the left-hand rest template. Calibration runs every session
because flex sensor resistance drifts with temperature, fatigue and how the
glove is worn.

To retrain from the recorded data:

```bash
python build_features.py
python train_classical.py --features rep_features.csv --outdir results
python train_words.py --no-deep
```

## Known limitations

The dataset was recorded by two signers and no signer identifier was stored,
so their repetitions are mixed. Both signers therefore appear in the training
split and in the held-out split, and the 99.21% above measures performance on
signers the model has already seen. Recording a signer column would allow a
leave-one-signer-out test, which this dataset cannot support as it stands.

Three people outside that pair did use the finished system, each running the
startup calibration first, and it responded to their signs. That is an
observation from the demo rather than a measured result: no per-letter tally
was kept, so it supports the claim that per-session calibration absorbs
differences between hands but it does not put a number on it.

Between 35% and 64% of flex samples sit at a normalisation rail. The calibrated
span is narrower than the range the fingers actually travel, so within-class
detail is lost even where classification succeeds.

The sensing is wearable but the processing is not. Training cannot run on the
ESP32 and inference would compete with the sampling loop, so both stay on the
host. LDA at 0.13 ms is small enough to port on-device later, which a recurrent
model would not have been.

Speech synthesis is the operating system voice through pyttsx3. No acoustic
model or vocoder was selected or trained. That was deliberate, since an
assistive device cannot depend on a network or tolerate a pause after every
sentence.
