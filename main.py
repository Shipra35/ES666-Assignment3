import pdb
import glob
import importlib.util
import os
import cv2

### Change path to images here
path = f'Images{os.sep}*'  # Use os.sep for cross-platform compatibility

# Get all submission folders inside ./src/
all_submissions = glob.glob('./src/*')

# Create results directory if it doesn't exist
os.makedirs('./results/', exist_ok=True)

for idx, algo in enumerate(all_submissions):
    print(
        f'****************\tRunning Awesome Stitcher developed by: '
        f'{algo.split(os.sep)[-1]}  | {idx + 1} of {len(all_submissions)}\t********************'
    )
    try:
        # Construct the path to stitcher.py
        module_name = f"{algo.split(os.sep)[-1]}_stitcher"
        filepath = f"{algo}{os.sep}stitcher.py"

        # Dynamically load the stitcher module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the PanaromaStitcher class from the loaded module
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        # Process each image set in the given path
        for impaths in glob.glob(path):
            print(f'\t\t Processing... {impaths}')

            # Run the panorama stitching method
            stitched_image, homography_matrix_list = inst.make_panaroma_for_images_in(path=impaths)

            # Construct the output path
            output_dir = f'./results/{impaths.split(os.sep)[-1]}'
            outfile = f'{output_dir}/{module_name}.png'

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save the stitched panorama
            cv2.imwrite(outfile, stitched_image)

            print(f'Homography Matrices:\n{homography_matrix_list}')
            print(f'Panorama saved ... @ {outfile}\n\n')

    except Exception as e:
        print(f'Oh No! My implementation encountered this issue\n\t{e}\n\n')
