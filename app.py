"""
app.py
This file acts as the main Flask server for the Explainable Interface for Neural Concept Binder (NCBXI).
It exposes multiple routes that handle image listing, model inference, and different inspection methods.
The logic here is tied closely to 'NCBXI_api.py', where model loading and processing functions (e.g., load_model, run_inference)
are defined and imported. The routes are triggered from the front-end (index.html + script.js) via POST requests,
and results or feedback are handled/stored accordingly.
"""

import logging
import os
import re
import torch
from flask import Flask, request, render_template, send_from_directory, jsonify, session, send_file
import pandas as pd
import pickle

# Import 'NCBXI_api.py' for model operations and concept inspections
from NCBXI_api import (
    Args, load_model, preprocess_image, run_inference, 
    visualize_block, implicit_inspection, comparative_inspection, 
    interventional_inspection, create_block_concepts, preprocess_image_paths
)

# Create a Flask application and specify the static folder.
# The static folder stores CSS, JS, and other assets used by index.html.
app = Flask(__name__, static_folder="static")

# A secret key is necessary for session data management (storing session variables, etc.).
# "NCBXI_SECRET_KEY" can be replaced with a secure random key.
app.secret_key = "NCBXI_SECRET_KEY"

# Paths to images, static plots, and feedback data.
# We define these so we can easily find where images and output plots are stored.
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # Go up one level from the directory
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "CLEVR-4-1", "test", "images")
PLOT_FOLDER = "static/images/plots/"

# Feedback folder/file where user feedback is stored in Excel format.
# The feedback is for collecting user annotations or corrections regarding concept blocks.
FEEDBACK_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "User Feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_FOLDER, "feedback.xlsx")

# Ensure feedback folder exists; create it if it doesn't already exist.
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

# --------------------------------------------------------------------------------------------------
# check_image_exists
# --------------------------------------------------------------------------------------------------
"""
Checks if the given image file exists in the IMAGE_FOLDER path.
image_path (str): Name of the image file to locate in the IMAGE_FOLDER.
returns bool: True if the image file is found, otherwise False.
"""
def check_image_exists(image_path):
    # Construct the full path to the image using IMAGE_FOLDER
    full_image_path = os.path.join(IMAGE_FOLDER, image_path)
    # If the file isn't found, log an error and return False
    if not os.path.exists(full_image_path):
        logging.error(f"ERROR: Image file not found at {full_image_path}")
        return False
    return True

# Ensure IMAGE_FOLDER exists (create it if missing).
# This ensures no errors occur when the code tries to save or retrieve images.
if not os.path.exists(IMAGE_FOLDER):
    print(f"Warning: Image folder '{IMAGE_FOLDER}' does not exist! Creating it now.")
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

# --------------------------------------------------------------------------------------------------
# get_sorted_images
# --------------------------------------------------------------------------------------------------
"""
Fetches all images (png, jpg, jpeg) from the IMAGE_FOLDER, for NCBXI we have only .png images
sorts images alphabetically, and prioritizes the specified image_path if provided.
image_path (str, optional): The name of the image to prioritize in the returned list.
returns list: Sorted list of image filenames, with the prioritized image (if any) at index 0.
"""
def get_sorted_images(image_path=None):
    # If the folder doesn't exist, provide a default fallback image list
    if not os.path.exists(IMAGE_FOLDER):
        print("Warning: Image folder does not exist! No images available.")
        return ["default.png"]

    try:
        # Collect all files that match image extensions
        images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        images.sort()
        
        # If a specific image_path was given and exists, put it at the front of the list
        if image_path and image_path in images:
            images.remove(image_path)
            images.insert(0, image_path)
        
        return images if images else ["default.png"]
    except FileNotFoundError:
        # If the folder is not found, return a single fallback image
        return ["default.png"]

# --------------------------------------------------------------------------------------------------
# preprocess_and_infer
# --------------------------------------------------------------------------------------------------
"""
Loads the model from NCBXI_api.py, preprocesses the specified image, 
and runs inference to obtain concept 'codes' from the neural binder.
image_path (str): Name of the image file to process and run inference on.
returns tuple: (codes, model_path, model, args, image_path)
 - codes: Model output, typically a PyTorch Tensor of concept IDs
 - model_path: Directory of the loaded model
 - model: The loaded PyTorch model instance
 - args: Config object specifying device, image size, and other parameters
 - image_path: Echo of the input parameter
"""
def preprocess_and_infer(image_path):
    full_image_path = os.path.join(IMAGE_FOLDER, image_path)
    
    # If the target image doesn't exist, skip further processing
    if not os.path.exists(full_image_path):
        print(f"Warning: Image '{full_image_path}' not found! Skipping inference.")
        return None, None, None, None, None  

    # Model path for loading the pre-trained neural concept binder
    model_path = "../model/CLEVR-4/retbind_seed_2/"
    args = Args(model_path=model_path)

    # Set manual seed for reproducibility (especially if the model requires a seed)
    torch.manual_seed(args.seed)
    device = args.device
    
    # Load model from NCBXI_api.py using the loaded Args
    model = load_model(args)
    
    # Convert the image from disk into a PyTorch tensor
    image_tensor, _ = preprocess_image(full_image_path, args.image_size)
    
    # Run inference to extract concept codes using NCBXI_api.py's run_inference
    codes, _ = run_inference(image_tensor, model, device)
    
    return codes, model_path, model, args, image_path

# --------------------------------------------------------------------------------------------------
# codes_to_string
# --------------------------------------------------------------------------------------------------
"""
Converts the inference output (PyTorch Tensor) into a string of integer concept IDs for readability.
codes (Tensor or None): The codes from model inference.
returns str or None: A stringified list of concept IDs, or None if codes is invalid.
"""
def codes_to_string(codes):
    # Return None if no codes are present
    if codes is None:
        return None
    raw_str = str(codes.cpu().numpy())
    # Use regex to extract numeric values from the raw tensor string
    activated_concepts = re.findall(r"[-+]?\d*\.\d+|\d+", raw_str)
    # Convert extracted strings to integers
    cleaned_activated_concepts = [int(num) for num in activated_concepts]
    return str(cleaned_activated_concepts)

# --------------------------------------------------------------------------------------------------
# home ("/")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/') Renders the main index.html template, showing a list of images (from get_sorted_images).
If POST request includes 'image_path', that image is prioritized for display.
returns HTML: Rendered page with image list for client interaction.
"""
@app.route("/", methods=["GET", "POST"])
def home():
    # If user submitted a form, retrieve the 'image_path' from it; otherwise None
    selected_image = request.form.get("image_path") if request.method == "POST" else None
    image_files = get_sorted_images(selected_image)

    # Render index.html with the sorted image list, 
    # the first image for display, and an empty plot_paths list
    return render_template(
        "index.html",
        image_list=image_files,
        first_image=image_files[0] if image_files else "default.png",
        image_path=selected_image,
        plot_paths=[]
    )

# --------------------------------------------------------------------------------------------------
# get_image ("/images/<filename>")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/images/<filename>') Retrieves and serves an image file from IMAGE_FOLDER for display.
filename (str): Name of the image to serve.
returns file: The image file if found, otherwise a Flask error/404 if missing.
"""
@app.route("/images/<filename>")
def get_image(filename):
    # send_from_directory will attempt to return the file named <filename> in IMAGE_FOLDER
    return send_from_directory(IMAGE_FOLDER, filename)

# --------------------------------------------------------------------------------------------------
# run_model ("/run_model")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/run_model') Accepts POST data with 'image_path', runs the model on that image,
then returns a JSON response containing device info, codes, and block concept counts.
This links to NCBXI_api.py for actual model operations.
returns JSON: { success, message, device, codes_str, concepts_per_block_str }
"""
@app.route("/run_model", methods=["POST"])
def run_model():
    from NCBXI_api import Args, create_block_concepts
    # Retrieve the image path from the form data
    image_path = request.form["image_path"]
    print(f"Received request to process image: {image_path}")

    # Run the entire inference pipeline (preprocessing, model forward pass)
    codes, model_path, model, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        print("Model inference failed or no output received!")
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    print("✅ Model inference successful, preparing response...")

    # Convert concept codes to a user-friendly string
    codes_str = codes_to_string(codes)

    # Build "Number of concepts per block" for the user interface
    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    concept_counts = []
    for blk_id, block_data in block_concepts.items():
        count = len(block_data["prototypes"]["ids"])
        concept_counts.append(count)
    concepts_per_block_str = str(concept_counts)

    # Create or increment a unique case ID for each model run, stored in the session
    if os.path.exists(FEEDBACK_FILE):
        df_existing = pd.read_excel(FEEDBACK_FILE)
        if not df_existing.empty:
            max_case = df_existing["Case"].max()
            current_case = int(max_case + 1)
        else:
            current_case = 1
    else:
        current_case = 1

    # Save info in the session to link feedback with the correct image/case
    session["case_id"] = current_case
    session["image_id"] = image_path
    session["codes_str"] = codes_str

    # Return a JSON object that script.js can use to display info to the user
    return jsonify({
        "success": True,
        "message": "Model run successful!",
        "device": args.device,
        "codes_str": codes_str,
        "concepts_per_block_str": concepts_per_block_str
    })

# --------------------------------------------------------------------------------------------------
# get_visualization ("/visualization")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/visualization') Expects 'image_path' and 'block_id'. Uses the concept binder to visualize
a specific block. The actual visualization logic is in NCBXI_api.py (visualize_block).
returns JSON: { success, message, plot_path, device, codes_str }
"""
@app.route("/visualization", methods=["POST"])
def get_visualization():
    # Extract fields from form data
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])

    # Check if the user-provided image is actually present
    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    # Re-run the model if needed to ensure codes are available
    codes, model_path, _, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    # visualize_block is an NCBXI_api.py function that generates a plot/image file for the concept block
    visualize_block(block_id, codes, model_path)
    codes_str = codes_to_string(codes)

    return jsonify({
        "success": True,
        "message": "Visualization completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Visualize_Concept_Block", "Concept_Block.png"),
        "device": args.device,
        "codes_str": codes_str
    })

# --------------------------------------------------------------------------------------------------
# get_implicit_inspection ("/implicit_inspection")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/implicit_inspection') Expects 'image_path', 'block_id', 'cluster_id'.
Shows training image examples that share a concept cluster with the user's chosen block/cluster.
returns JSON: { success, message, plot_path, device, codes_str }
"""
@app.route("/implicit_inspection", methods=["POST"])
def get_implicit_inspection():
    from NCBXI_api import create_block_concepts, preprocess_image_paths
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])
    cluster_id = int(request.form["cluster_id"])

    # Verify image presence
    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    # Inference again to get codes and device info
    codes, model_path, _, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    codes_str = codes_to_string(codes)

    # create_block_concepts returns a dictionary of block => concept prototypes
    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    if block_id not in block_concepts:
        msg = (f"Block ID {block_id} is out of range. "
               f"Available blocks: {list(block_concepts.keys())}.")
        return jsonify({"success": False, "message": msg}), 200

    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        msg = (f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
               f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return jsonify({"success": False, "message": msg}), 200

    # Pre-process image path
    with open("../model/CLEVR-4/retbind_seed_2/all_img_locs.pkl", "rb") as f:
        all_img_locs = pickle.load(f)

    all_img_locs = preprocess_image_paths(all_img_locs) 

    # The actual inspection is performed in NCBXI_api.py
    implicit_inspection(block_concepts, all_img_locs, block_id=block_id, cluster_id=cluster_id)
    return jsonify({
        "success": True,
        "message": "Implicit Inspection completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Implicit_Inspection", "Implicit_Inspection.png"),
        "device": args.device,
        "codes_str": codes_str
    })

# --------------------------------------------------------------------------------------------------
# get_comparative_inspection ("/comparative_inspection")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/comparative_inspection') Expects 'image_path', 'block_id', 'cluster_id', 'num_exemplars'.
Collects training images for the same cluster concept, comparing them with the user’s image.
returns JSON: { success, message, plot_path, device, codes_str }
"""
@app.route("/comparative_inspection", methods=["POST"])
def get_comparative_inspection():
    from NCBXI_api import create_block_concepts, preprocess_image_paths
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])
    cluster_id = int(request.form["cluster_id"])
    num_exemplars = int(request.form["num_exemplars"])

    # Validate the user's requested image
    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    # Ensure the number of exemplars is within acceptable bounds
    if num_exemplars < 1 or num_exemplars > 6:
        return jsonify({
            "success": False,
            "message": "Number of exampler should be between 1 to 6"
        }), 200

    codes, model_path, model, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    codes_str = codes_to_string(codes)

    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    if block_id not in block_concepts:
        msg = (f"Block ID {block_id} is out of range. "
               f"Available blocks: {list(block_concepts.keys())}.")
        return jsonify({"success": False, "message": msg}), 200

    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        msg = (f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
               f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return jsonify({"success": False, "message": msg}), 200

    # Pre-process image path
    with open("../model/CLEVR-4/retbind_seed_2/all_img_locs.pkl", "rb") as f:
        all_img_locs = pickle.load(f)

    all_img_locs = preprocess_image_paths(all_img_locs) 

    # The comparative_inspection function in NCBXI_api.py handles the actual analysis and plotting
    comparative_inspection(
        block_concepts,
        all_img_locs,
        model,
        example_path=os.path.join(IMAGE_FOLDER, image_path),
        block_id=block_id,
        cluster_id=cluster_id,
        num_exemplars=num_exemplars
    )

    return jsonify({
        "success": True,
        "message": "Comparative Inspection completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Comparative_Inspection", "Comparative_Inspection.png"),
        "device": args.device,
        "codes_str": codes_str
    })

# --------------------------------------------------------------------------------------------------
# get_interventional_inspection ("/interventional_inspection")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/interventional_inspection') Expects 'image_path', 'block_id', 'cluster_id'.
Temporarily alters the concept in one block and re-runs inference to see model response.
returns JSON: { success, message, plot_path, device, codes_str }
"""
@app.route("/interventional_inspection", methods=["POST"])
def get_interventional_inspection():
    from NCBXI_api import create_block_concepts
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])
    cluster_id = int(request.form["cluster_id"])

    # Confirm the image's existence
    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    # Gather the codes to confirm the block is relevant
    codes, model_path, model, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    codes_str = codes_to_string(codes)

    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    if block_id not in block_concepts:
        msg = (f"Block ID {block_id} is out of range. "
               f"Available blocks: {list(block_concepts.keys())}.")
        return jsonify({"success": False, "message": msg}), 200

    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        msg = (f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
               f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return jsonify({"success": False, "message": msg}), 200

    # The interventional_inspection modifies model internals for the specified concept, 
    # generating a new plot to illustrate differences
    interventional_inspection(
        block_concepts,
        model,
        example_path=os.path.join(IMAGE_FOLDER, image_path),
        block_id=block_id,
        cluster_id=cluster_id,
        args=args
    )
    return jsonify({
        "success": True,
        "message": "Interventional Inspection completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Interventional_Inspection", "Interventional_Inspection.png"),
        "device": args.device,
        "codes_str": codes_str
    })

# --------------------------------------------------------------------------------------------------
# save_feedback ("/save_feedback")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/save_feedback') Expects 'block_id' and 'feedback_label'.
Records user feedback (labels per concept block) in feedback.xlsx.
returns JSON: success status + message
"""
@app.route("/save_feedback", methods=["POST"])
def save_feedback():
    block_id = request.form.get("block_id", None)
    feedback_label = request.form.get("feedback_label", "")

    case_id = session.get("case_id", 0)
    image_id = session.get("image_id", "")
    codes_str = session.get("codes_str", "")

    if block_id is None:
        return jsonify({"success": False, "message": "No block_id specified."}), 400
    try:
        block_id = int(block_id)
    except ValueError:
        return jsonify({"success": False, "message": "block_id must be an integer."}), 400

    columns = (
        ["Case", "Image ID", "Activated Concepts for Each Block"]
        + [f"Block {i}" for i in range(16)]
    )

    # If the feedback file doesn't exist, a feedback.xlsx will be generated
    if not os.path.exists(FEEDBACK_FILE):
        df_blank = pd.DataFrame(columns=columns)
        df_blank.to_excel(FEEDBACK_FILE, index=False)

    df_existing = pd.read_excel(FEEDBACK_FILE)

    # If no rows exist yet, create a brand-new row
    if df_existing.empty:
        row_data = {
            "Case": [case_id],
            "Image ID": [image_id],
            "Activated Concepts for Each Block": [codes_str],
        }
        for i in range(16):
            row_data[f"Block {i}"] = [""]
        row_data[f"Block {block_id}"] = [feedback_label]
        df_new = pd.DataFrame(row_data, columns=columns)
        df_result = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # Attempt to match an existing row for the case+image combination
        match_mask = (df_existing["Case"] == case_id) & (df_existing["Image ID"] == image_id)
        if match_mask.any():
            df_existing.loc[match_mask, f"Block {block_id}"] = feedback_label
            df_result = df_existing
        else:
            row_data = {
                "Case": [case_id],
                "Image ID": [image_id],
                "Activated Concepts for Each Block": [codes_str],
            }
            for i in range(16):
                row_data[f"Block {i}"] = [""]
            row_data[f"Block {block_id}"] = [feedback_label]
            df_new = pd.DataFrame(row_data, columns=columns)
            df_result = pd.concat([df_existing, df_new], ignore_index=True)

    # Overwrite or create the Excel file with updated feedback
    df_result.to_excel(FEEDBACK_FILE, index=False)

    return jsonify({"success": True, "message": f"Feedback saved for block id {block_id}."})

# --------------------------------------------------------------------------------------------------
# download_feedback_file ("/download_feedback")
# --------------------------------------------------------------------------------------------------
"""
(Route: '/download_feedback') Allows users to download the feedback.xlsx file directly
if it exists on the server. If not, returns a simple 404 text response.
"""
@app.route("/download_feedback", methods=["GET"])
def download_feedback_file():
    # Provide the file as an attachment if it exists
    if os.path.exists(FEEDBACK_FILE):
        return send_file(FEEDBACK_FILE, as_attachment=True)
    else:
        return "No feedback file found", 404

# --------------------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Start Flask in debug mode. 
    # This launches a development server at http://0.0.0.0:5000/
    app.run(host="0.0.0.0", port=5000, debug=True)
