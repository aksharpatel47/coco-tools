import traitlets
from ipywidgets import widgets
from IPython.display import display
from tkinter import Tk, filedialog
import os
from tools.image import InferImage
from tools.tf_record import write_inference_record
import glob
import datetime
from subprocess import check_call


class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select Directory!!"
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        b.value = filedialog.askdirectory()
        print(b.value)
        image_path_list = glob.glob(os.path.join(b.value, "**", "*.jpg"))
        print("Images found: ", len(image_path_list))
        b.description = "Directory Selected!!!"
        b.icon = "check-square-o"
        b.style.button_color = "lightgreen"


model_widget = widgets.Dropdown(
    options=['crosswalk', 'crosswalk-60', 'crosswalk-629', 'walksignal', 'walksignal-60'], value='crosswalk')
dir_button = SelectFilesButton()
infer_button = widgets.Button(description="Run Inference")


def on_inference_clicked(b):
    dir_path = dir_button.value
    print("Looking through the following directory")
    print("="*50)
    print(dir_path)
    print("="*50)
    image_lookup_path = os.path.join(dir_path, "**", "*.jpg")
    print("Looking in: ", image_lookup_path)
    image_path_list = glob.glob(image_lookup_path)
    images = [InferImage(p) for p in image_path_list]
    print("Number of Images: ", len(images))
    write_inference_record("infer", images)

    job_name = "{}_{}_inference_{}".format(os.path.basename(dir_path), model_widget.value, datetime.datetime.today(
    ).isoformat()).replace("-", "_").replace(":", "_").replace(".", "_")
    job_dir = "gs://gmaps-mini/infer"
    input_record = f"gs://gmaps-mini/infer/inputs/{job_name}.record"
    output_record = "gs://gmaps-mini/infer/results/{}.record".format(job_name)
    inference_model = "gs://gmaps-mini/models/{}.pb".format(model_widget.value)

    print("Uploading to: " + input_record)
    up_str = f"gsutil cp infer.record {input_record}"

    check_call(up_str, shell=True)

    print("Starting Inference")
    shell_str = f"gcloud ml-engine jobs submit training {job_name} --runtime-version 1.13 --job-dir={job_dir} --packages packages/object_detection-0.1.tar.gz,packages/slim-0.1.tar.gz,packages/pycocotools-2.0.zip --module-name object_detection.inference.infer_detections --region us-central1 --config cloud.yml -- --input_tfrecord_paths={input_record} --output_tfrecord_path={output_record} --inference_graph={inference_model} --discard_image_pixels=True"
    check_call(shell_str, shell=True)


infer_button.on_click(on_inference_clicked)


def get_infer_widget():
    vbox = widgets.VBox([model_widget, dir_button, infer_button])
    return vbox
