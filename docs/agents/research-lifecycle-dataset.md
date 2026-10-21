# Research: best dataset for the lifecycle loop

Ticket: [#651](https://github.com/SkafteNicki/dtu_mlops/issues/651), part of map
[#647](https://github.com/SkafteNicki/dtu_mlops/issues/647). Researched 2026-10-21.

## Answer

**Recommend: Rock-Paper-Scissors (Laurence Moroney, CGI hands).** Students photograph their own hands with their
phones. Labeling takes about one second per image, and nobody disagrees about the label. The training set is
CGI-rendered hands on a white background, so real phone photos drift strongly and visibly from it, which suits the
monitoring day. A small CNN trained from scratch reaches 94-99% test accuracy on a laptop CPU in under a minute
(measured below). The licence is CC BY 2.0 and the download is 230 MB.

**Runner-up: keep (corrupted) MNIST, adding digits drawn in the frontend or handwritten on paper and photographed.**
This costs almost nothing to migrate and trains in seconds. Its weaknesses are that drawn digits drift only mildly and
that a phone photo of a digit needs a preprocessing step before it looks like MNIST. QuickDraw is a close third, with
the same frontend-drawing story and more classes.

**EuroSAT ([#564](https://github.com/SkafteNicki/dtu_mlops/issues/564)) fits the loop badly.** Students cannot capture
new satellite tiles, and labeling land-use tiles is slow and ambiguous. It is a fine dataset for training, but it
cannot close the loop.

## Criteria and scoring

Each criterion is scored 1-5, with 5 the best.

1. **Phone**: 5 means students can photograph new examples with their phones anywhere, including indoors in January.
   3 means new data can only come from drawing in the frontend. 1 means students cannot produce new data at all.
2. **Label**: how fast and unambiguous labeling is for a student with no domain knowledge.
3. **Drift**: how natural and visible the gap is between the training data and the data students capture.
4. **CPU**: whether the model trains to reasonable accuracy on a laptop CPU in minutes. The table notes whether that
   means training from scratch or fine-tuning a small pretrained model.
5. **Licence, size, loading**: a permissive licence, a small download, and loading through torchvision or HF
   `datasets`.

| Candidate | Phone | Label | Drift | CPU | Lic/size/load | **Total** |
| --- | --- | --- | --- | --- | --- | --- |
| **Rock-Paper-Scissors (CGI)** | 5 | 5 | 5 | 5 (scratch) | 4 | **24** |
| Corrupted MNIST + drawn/photographed digits (baseline) | 3 | 5 | 3 | 5 (scratch) | 5 | **21** |
| QuickDraw (subset of classes) | 3 | 4 | 4 | 5 (scratch) | 4 | **20** |
| Sign Language MNIST | 4 | 2 | 4 | 5 (scratch) | 4 | **19** |
| Fruits-360 (subset of classes) | 4 | 3 | 5 | 4 (scratch at 100 px) | 3 | **19** |
| CIFAR-10 | 2 | 4 | 4 | 5 (scratch) | 4 | **19** |
| HaGRID (30k 384p sample) | 5 | 4 | 4 | 3 (fine-tune) | 2 | **18** |
| TrashNet | 4 | 3 | 4 | 3 (fine-tune) | 3 | **17** |
| RealWaste | 3 | 3 | 5 | 3 (fine-tune) | 3 | **17** |
| CORe50 (10 categories) | 4 | 5 | 4 | 3 (fine-tune) | 1 | **17** |
| Imagenette (160 px) | 1 | 4 | 3 | 3 (fine-tune) | 4 | **15** |
| EuroSAT RGB (#564) | 1 | 2 | 3 | 3 (scratch, slower) | 5 | **14** |

I ruled out the following without scoring them. Food-101 is a 5.0 GB download, and the source does not state a clear
licence. Flowers102 and plant or leaf datasets such as PlantVillage depend on the season: the course runs in a Danish
January. They also need botanical or disease knowledge to label. DTD textures has ambiguous labels and is 625 MB. LEGO
bricks and coins assume students own the objects, and their labels are fine-grained.

## Candidate notes

### Rock-Paper-Scissors (recommended)

- **Source and licence**: CGI images of hands of "various races, ages and genders" on a white background, 300x300 RGB,
  licensed **CC BY 2.0**, which allows commercial use and requires attribution
  ([laurencemoroney.com/datasets.html](https://laurencemoroney.com/datasets.html)). The author made them specifically
  "to test whether a model trained on CGI data could classify real images". That is exactly the drift question for the
  monitoring day.
- **Size and splits**: 2,520 training and 372 test images, 3 balanced classes
  ([TFDS catalog](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)). Download sizes, verified with
  HEAD requests: `rps.zip` is 200.7 MB and `rps-test-set.zip` is 29.5 MB, both at
  `storage.googleapis.com/download.tensorflow.org/data/`. The `rps-validation.zip` URL that is sometimes referenced
  returns 403. I checked a sample, and the test set is also CGI, so it measures in-distribution accuracy only.
- **Loading**: torchvision has no built-in class, but the unzipped folders are already in `ImageFolder` layout
  (`rps/{rock,paper,scissors}/*.png`). It is also in TFDS, and there is an HF mirror at `jerpint/rock-paper-scissors`
  (cc-by-2.0, contains the zips). A small download-and-unpack step is a natural first ingest exercise for Day 4.
- **Phone capture and labeling**: students need only their hands. Photos show hands, not faces, which keeps GDPR
  concerns low. EXIF metadata still has to be stripped on upload. There are 3 obvious classes, so labeling takes about
  a second per image and a labeling-tool exercise can review many images quickly.
- **Drift**: the gap from CGI on a white background to real skin, real backgrounds, lighting, sleeves and phone
  aspect ratios is large and visible. Simple monitoring statistics such as brightness, background colour and
  embedding distance should detect it. Accuracy on real photos is expected to drop. I have **not verified** how far,
  because I had no real photos. A 30-photo prototype would settle it.
- **Room for a second pass**: the second pass of the loop could introduce a fourth class, such as "none/other" or
  "lizard", to show a label-schema change. This is only a suggestion.
- **Weaknesses**: with only 3 classes, the in-distribution task is almost too easy. That is acceptable, because the
  point of the loop is the real-photo gap. Attribution must be kept. The download is 230 MB, compared with 10 MB for
  MNIST.

### Corrupted MNIST + frontend digits (baseline)

- MNIST is 9.9 MB for the training images (`ossci-datasets.s3.amazonaws.com/mnist/`, the torchvision mirror) and is
  built into torchvision. The original page states no explicit licence. The HF card `ylecun/mnist` tags it MIT.
- New data can come from a canvas in the frontend, or from a phone photo of a digit handwritten on paper. The paper
  photo needs thresholding, cropping and centring, which is a useful pipeline step to teach. Labeling is trivial.
- Drift from canvas drawings is mild: stroke width and centring change. Paper photos drift more. Migration costs
  nothing, since everything already uses it. The course's own motivation in #564 ("more engaging than rotated digits")
  argues against keeping it.

### Other candidates

- **QuickDraw**: CC BY 4.0, 345 categories, 28x28 `.npy` bitmaps
  ([repo](https://github.com/googlecreativelab/quickdraw-dataset)). Each category is about 97 MB (`cat.npy` measured),
  so a 10-class subset is about 1 GB unless it is subsampled. New data comes from frontend drawing, not phone photos.
  Drawing with a mouse or finger instead of the original input gives moderate drift.
- **Sign Language MNIST**: CC0, 27,455 training and 7,172 test images, 28x28 grayscale, 24 letters (J and Z are
  excluded because they need motion), 106 MB on Kaggle
  ([Kaggle API metadata](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)). It is phone-friendly, but
  labelers have to know ASL letters, so labeling is slow and error-prone. Kaggle downloads require a login.
- **Fruits-360**: CC BY-SA 4.0 (share-alike), 196,522 images in 274 classes, 100x100, filmed on a rotating motor
  against a white background ([repo](https://github.com/fruits-360/fruits-360-100x100)). Drift to a kitchen-table phone
  photo is great, but students need the produce, the variety-level classes are ambiguous, and Kaggle holds 8.7 GB
  across all versions.
- **CIFAR-10**: 170 MB and built into torchvision. The source states no explicit licence. Only a few classes (car,
  maybe cat or bird) can be photographed in Copenhagen in January.
- **HaGRID**: 33 gestures, about 1.09M FullHD images. The full set is 1.5 TB and the 512p version is 119.4 GB. The
  licence is a variant of CC BY-SA 4.0. The images show whole people from 65,977 subjects, including faces
  ([repo](https://github.com/hukenovs/hagrid)). Community samples on HF (e.g. `cj-mills/hagrid-sample-30k-384p`) are
  smaller. Licence, size and privacy all count against it, and it needs a pretrained backbone.
- **TrashNet**: MIT, 2,527 images in 6 classes, shot on a white posterboard with iPhones. The original is 3.5 GB and
  hosted on Google Drive ([repo](https://github.com/garythung/trashnet)). The HF mirror `garythung/trashnet` is
  3.68 GB with 5,054 rows, which looks duplicated. "Trash" versus "plastic" is ambiguous, and students need actual
  waste.
- **RealWaste**: CC BY 4.0, 4,752 images in 9 classes, 524x524, photographed in a real landfill. It is a 656.6 MB zip
  from UCI ([UCI 908](https://archive.ics.uci.edu/dataset/908/realwaste)). Drift is strong, but the classes overlap
  (miscellaneous trash, textile, food).
- **CORe50**: CC BY 4.0 according to the project page. Its 10 everyday categories (plug adapters, phones, scissors,
  light bulbs, cans, glasses, balls, markers, cups, remote controls) are ideal for phone photos
  ([site](https://vlomonaco.github.io/core50/)). However, the 128x128 zip is **5.9 GB** (measured), which rules it out.
- **Imagenette**: Apache-2.0 for the fastai subset ([repo](https://github.com/fastai/imagenette)), but the images come
  from ImageNet. The 160 px version is 99 MB and it is built into torchvision. Its classes (tench, French horn,
  parachute, church...) cannot be photographed by students.
- **EuroSAT RGB**: MIT, 27,000 images in 10 classes, 64x64. `EuroSAT_RGB.zip` is 94.7 MB and the multispectral
  version is 2.1 GB ([Zenodo 7711810](https://zenodo.org/record/7711810), [repo](https://github.com/phelber/eurosat)).
  It is built into torchvision, which makes it excellent for criterion 5. For the loop it fails: new data would have to
  come from Sentinel-2 downloads, and labeling land use (for example PermanentCrop against AnnualCrop) needs domain
  knowledge. The #564 plan's use cases for data engineering (band pipelines, validation) still apply if the course ever
  wants them.

## Evidence: CPU training check for Rock-Paper-Scissors

I ran a throwaway script outside the repo: `uv run`, torch CPU, 4 threads, 20-core desktop. It trained on the
2,520-image training set and evaluated on the 372-image CGI test set.

| Model | Input | Epochs | Test acc (per epoch) | Wall time |
| --- | --- | --- | --- | --- |
| 3-layer CNN from scratch | 64x64 | 5 | 0.76, 0.93, 0.99, 0.94, 0.94 | 42 s |
| MobileNetV3-small (ImageNet), only the last layer trained | 128x128 | 3 | 0.66, 0.68, 0.76 | 19 s |

A small CNN trained from scratch is enough, so no pretrained model is needed. The noisy epochs suggest adding
augmentation and a validation split. Training only the last layer of a pretrained model did worse here. I did not
measure the drop on real phone photos.

## Open questions for the grilling tickets

- Should the course capture photos at a fixed size and aspect ratio in the frontend (for example a square crop), or
  leave aspect-ratio drift in as a monitoring signal?
- Should the course keep the "corrupted" idea, for example by adding synthetic corruptions to RPS on Day 4, or drop it?
- Attribution text for CC BY 2.0 needs to go into the course pages and the dataset card.
