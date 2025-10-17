# Mechatronics_Vision

## Setup
1. Make sure on the root directory you have the following files:
```yaml
data:
    images:
        train:
            <all photos here>
    labels:
        train:
            <all labels here>
```
2. In `config.yaml` make sure to set `path:` to your data folder's absolute directory. (relative directory is sometimes buggy for this)

## Documentation
### file_integrity.py
1. Checks if you have the proper directories, i.e.: data and the corresponding images/labels folder which must be spelled exactly as 'images' and 'labels'
2. Verifies your files in the 'data' directory match up, for YOLO to work you need photos and text files in seperate folders that have identifical names. the annotations in the `.txt` file line up in the image in your `.png` (or other image extension) files by the name of the file
### train.py 
1. This trains your YOLO-model using the information from your data directory
2. The YOLO model will then be saved as a `best.pt` file
### source.py
1. For configuring your source (should be from Zed camera but always you to use your webcam for validation)
2. This also is for choosing (and renaming) your `.pt` file that you're using
### run.py
1. For running your model and returning necessary information to shared memory
2. run.py relies on `source.py` to be configured to the ZED camera and your `.pt` model.
## Please Note:
### This issue pertains to Summer 2025 data:
1. The following images won't work due to their annotation being off bound, you can choose to force their annotations to `1 0.5 0.5 1 1` or delete them entirely.
> 0475, 0512, 1153, 1240, 1684, 1788, 1995, 2123, 2904, 2959, 3656, 3690, 4630

feel free to use the terminal code below to move them into a `ignore` file instead (create the folder under 'data/images' and 'data/labels' first)
```bash
mv Mechatronics_Vision_2026/data/images/train/screenshot_0475.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_0512.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_1153.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_1240.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_1684.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_1788.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_1995.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_2123.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_2904.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_2959.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_3656.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_3690.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/images/train/screenshot_4630.png Mechatronics_Vision_2026/data/images/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_0475.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_0512.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_1153.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_1240.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_1684.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_1788.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_1995.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_2123.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_2904.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_2959.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_3656.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_3690.txt Mechatronics_Vision_2026/data/labels/ignore
mv Mechatronics_Vision_2026/data/labels/train/screenshot_4630.txt Mechatronics_Vision_2026/data/labels/ignore
```
2. The following `screenshot.txt` files are randomly duplicated:
> 'screenshot_0102(1)'. 'screenshot_0467(1)', 'screenshot_0473(1)', 'screenshot_0499(1)', 'screenshot_0664(1)', 'screenshot_0856(1)', 'screenshot_1023(1)', 'screenshot_1023(2)', 'screenshot_1234(1)', 'screenshot_1590(1)', 'screenshot_1751(1)', 'screenshot_1779(1)', 'screenshot_1787(1)', 'screenshot_1793(1)', 'screenshot_1989(1)', 'screenshot_2073(1)', 'screenshot_2099(1)', 'screenshot_2258(1)', 'screenshot_2264(1)', 'screenshot_2265(1)', 'screenshot_2270(1)', 'screenshot_2502(1)', 'screenshot_2715(1)', 'screenshot_3146(1)', 'screenshot_3190(1)', 'screenshot_3387(1)', 'screenshot_3423(1)', 'screenshot_3634(1)', 'screenshot_4173(1)', 'screenshot_4173(2)', 'screenshot_4199(1)', 'screenshot_4364(1)', 'screenshot_4402(1)', 'screenshot_4416(1)', 'screenshot_4601(1)', 'screenshot_4614(1)', 'screenshot_4615(1)', 'screenshot_4615(2)', 'screenshot_4629(1)', 'screenshot_4827(1)', 'screenshot_5052(1)', 'screenshot_5090(1)', 'screenshot_5245(1)', 'screenshot_5245(2)', 'screenshot_5279(1)', 'screenshot_5287(1)', 'screenshot_5293(1)', 'screenshot_5537(1)', 'screenshot_5906(1)', 'screenshot_5912(1)'

Here's a bash script to quickly delete all of these
```bash
rm Mechatronics_Vision_2026/data/labels/train/screenshot_0102(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_0467(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_0473(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_0499(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_0664(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_0856(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1023(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1023(2).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1234(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1590(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1751(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1779(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1787(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1793(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_1989(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2073(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2099(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2258(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2264(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2265(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2270(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2502(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_2715(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_3146(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_3190(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_3387(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_4614(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_4615(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_4615(2).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_4629(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_4827(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5052(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5090(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5245(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5245(2).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5279(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5287(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5293(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5537(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5906(1).txt
rm Mechatronics_Vision_2026/data/labels/train/screenshot_5912(1).txt
```