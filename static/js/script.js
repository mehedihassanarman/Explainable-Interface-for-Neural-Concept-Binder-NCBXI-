// script.js
// This file controls the front-end interactivity for NCBXI's index.html.
// It handles dynamic image loading, form submissions to app.py's endpoints (e.g., run_model, visualization),
// manages feedback inputs, toggles UI sections, and shows/hides modals.
//
// Linked to:
// - index.html, which references this script at the bottom of the page.
// - styles.css, which defines styles for the modals, buttons, spinners, and more.
// - app.py, as the AJAX (fetch) calls submit data to Flask routes that perform model operations.
//
// Wait for the entire HTML DOM to load before running any script
document.addEventListener("DOMContentLoaded", function () {
    // Retrieve the list of image filenames from the hidden JSON script tags in index.html
    let images = JSON.parse(document.getElementById("image-data").textContent);
    let firstImage = JSON.parse(document.getElementById("first-image").textContent);

    // If there's no valid "firstImage", use the first in the images array if available, or default to "default.png"
    if (!firstImage || firstImage === "undefined") {
        firstImage = images.length > 0 ? images[0] : "default.png";
    }

    // Locate the index of the initially selected image in the images array
    let index = images.indexOf(firstImage);
    if (index === -1) index = 0;

    // Grab references to key DOM elements
    let displayImage = document.getElementById('display-image');
    let selectedImageName = document.getElementById('selected-image-name');
    let galleryContainer = document.getElementById("gallery-container");
    let plotContainer = document.querySelector(".right"); 

    let deviceLine = document.getElementById("device-line");
    let conceptLine = document.getElementById("concept-line"); // ADDED p element
    let codesLine = document.getElementById("codes-line");

    // Create a spinner element to visually indicate loading state during lengthy operations
    let spinner = document.createElement("div");
    spinner.classList.add("spinner");
    spinner.innerHTML = `<div class="spinner-circle"></div><p>Loading analysis...</p>`;

    // Track if any analysis (model run, visualization, inspection, etc.) is currently running
    window.isAnalysisInProgress = false;

    // ADDED: Track if we are currently typing the device/concepts/codes lines.
    window.isTypingModelInfo = false; // <--- ADDED

    // Flags and variables for visualization states, feedback, and model run status
    window.isVisualizationPlot = false;
    window.currentBlockId = null;
    window.modelHasRun = false;

    window.lastImageWithModelInfo = null;
    window.lastPressedButton = null;
    window.isFeedbackOpen = false;

    // Store bracketed "concepts_per_block_str" from model results
    window.lastNumberOfConcepts = null;

    // ------------------------------------------------------------------
    // Optional: If you want a function to visually disable the gallery, 
    // you can re-use your existing setGalleryDisabled code
    // ------------------------------------------------------------------

    function setGalleryDisabled(disabled) {
        const imageName = document.getElementById("selected-image-name");
        const prevBtn = document.querySelector(".nav-button.prev");
        const nextBtn = document.querySelector(".nav-button.next");

        if (!imageName || !prevBtn || !nextBtn) return;

        if (disabled) {
            imageName.classList.add("disabled-gallery");
            prevBtn.classList.add("disabled-gallery");
            nextBtn.classList.add("disabled-gallery");
        } else {
            imageName.classList.remove("disabled-gallery");
            prevBtn.classList.remove("disabled-gallery");
            nextBtn.classList.remove("disabled-gallery");
        }
    }

    // Disable all numeric inputs while an analysis is running
    function disableAnalysisInputs() {
        document.querySelectorAll('input[name="block_id"], input[name="cluster_id"], input[name="num_exemplars"]').forEach(el => {
            el.disabled = true;
        });
    }
    // Re-enable the numeric inputs once analysis completes
    function enableAnalysisInputs() {
        document.querySelectorAll('input[name="block_id"], input[name="cluster_id"], input[name="num_exemplars"]').forEach(el => {
            el.disabled = false;
        });
    }
	
    // Visually indicate which button is currently selected by the user
    function highlightButton(btn) {
        if (window.lastPressedButton && window.lastPressedButton !== btn) {
            window.lastPressedButton.classList.remove("selected-button");
        }
        btn.classList.add("selected-button");
        window.lastPressedButton = btn;
    }

    // Initialize click listeners for all buttons that carry the "selectable-button" class
    function initSelectableButtons() {
        let selectableButtons = document.querySelectorAll(".selectable-button");
        selectableButtons.forEach(b => {
            b.addEventListener("click", function(evt) {
                highlightButton(evt.currentTarget);
            });
        });
    }

    // Globally disable all buttons on the page to prevent multiple operations from conflicting
    function disableAllButtons() {
        document.querySelectorAll("button").forEach(btn => {
            btn.disabled = true;
        });
    }
    // Enable all buttons on the page after an operation completes
    function enableAllButtons() {
        document.querySelectorAll("button").forEach(btn => {
            btn.disabled = false;
        });
    }

    // Clears out any existing plots from the output section
    function clearPlotImages() {
        plotContainer.innerHTML = `
          <div class="modern-box">
            <h2 class="modern-subtitle">Output Section</h2>
          </div>
          <p style="font-family:'Comic Sans MS'; font-weight:bold; font-size:16px;">
            No Plot Available for Visualization.
          </p>
        `;
        console.log("🧹 Cleared Output Section");
    }

    // Displays the spinner on the page and disables inputs/buttons
    function showSpinner() {
        plotContainer.appendChild(spinner);
        disableAllButtons();
        disableAnalysisInputs();
        window.isAnalysisInProgress = true;
    }
    // Removes the spinner and re-enables inputs/buttons
    function hideSpinner() {
        if (plotContainer.contains(spinner)) {
            plotContainer.removeChild(spinner);
        }
        enableAllButtons();
        enableAnalysisInputs();
        window.isAnalysisInProgress = false;
    }

    // Updates the Output Section with a newly fetched or generated plot image
    function updatePlotImage(plotPath) {
        plotContainer.innerHTML = `
          <div class="modern-box">
            <h2 class="modern-subtitle">Output Section</h2>
          </div>
        `;
        let img = document.createElement("img");
        img.src = `${plotPath}?t=${new Date().getTime()}`;
        img.alt = "Generated Plot";
        img.className = "plot-image";

        // Clicking the plot image opens a modal with a larger view
        img.addEventListener("click", () => {
            openPlotModal(img.src);
        });

        plotContainer.appendChild(img);
        console.log(`📈 Updated Plot: ${plotPath}`);
    }
	
    // Displays a typing animation for error or info messages in the output section
    function typePlotMessage(text) {
        plotContainer.innerHTML = `
          <div class="modern-box">
            <h2 class="modern-subtitle">Output Section</h2>
          </div>
        `;
        let spacer = document.createElement("div");
        spacer.style.height = "20px";
        plotContainer.appendChild(spacer);

        let messageElem = document.createElement("p");
        messageElem.style.fontFamily = "'Comic Sans MS', sans-serif";
        messageElem.style.fontSize = "16px";
        messageElem.style.fontWeight = "bold";
        messageElem.style.color = "#B71C1C"; 
        plotContainer.appendChild(messageElem);

        let i = 0;
        function typeChar() {
            if (i < text.length) {
                messageElem.textContent += text.charAt(i);
                i++;
                setTimeout(typeChar, 30);
            }
        }
        typeChar();
    }

    // Displays a typed message specifically for feedback operations at the bottom of the output section
    function typeFeedbackMessage(text, color = "#000") {
        let oldMsg = document.getElementById("feedback-message-typed");
        if (oldMsg) {
            oldMsg.remove();
        }
        let messageElem = document.createElement("p");
        messageElem.id = "feedback-message-typed";
        messageElem.style.fontFamily = "'Comic Sans MS'";
        messageElem.style.fontSize = "16px";
        messageElem.style.fontWeight = "bold";
        messageElem.style.marginTop = "10px";
        messageElem.style.color = color; 
        plotContainer.appendChild(messageElem);

        let i = 0;
        function typeChar() {
            if (i < text.length) {
                messageElem.textContent += text.charAt(i);
                i++;
                setTimeout(typeChar, 30);
            }
        }
        typeChar();
    }

    // Hide the model/device info lines
    function hideModelInfo() {
        deviceLine.style.display = "none";
        conceptLine.style.display = "none";
        codesLine.style.display = "none";

        deviceLine.textContent = "";
        conceptLine.textContent = "";
        codesLine.textContent = "";
    }

    // Utility function to display text in a typing animation
    function typeText(element, text, callback) {
        element.style.display = "block";
        element.textContent = "";
        let i = 0;
        let speed = 30;
        function typeChar() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(typeChar, speed);
            } else {
                if (callback) callback();
            }
        }
        typeChar();
    }

    // Manage the display of device and codes lines in the #model-info-box with optional typed animation
    function showModelInfo(device, codesStr, skipCheck, completionCallback) {
        if (!completionCallback) completionCallback = () => {};

        // ADDED: We are about to type device and concept lines => block image changes until done
        window.isTypingModelInfo = true; // <--- ADDED

        // skipCheck avoids re-displaying same info if it's identical
        if (!skipCheck && window.lastImageWithModelInfo === device + codesStr) {
            console.log("Skipping device/codes re-display for same details.");
            // ADDED: If we skip, we are effectively done typing
            window.isTypingModelInfo = false; // <--- ADDED
            completionCallback();
            return;
        }

        window.lastImageWithModelInfo = device + codesStr;

        disableAllButtons();
        disableAnalysisInputs();

        function doneTyping() {
            enableAllButtons();
            enableAnalysisInputs();
            window.isAnalysisInProgress = false;

            // ADDED: We have finished typing => user can switch images again
            window.isTypingModelInfo = false; // <--- ADDED

            completionCallback();
        }

        // Type out device line => concept line => codes line sequentially
        if (device) {
            let devText = `Device on: ${device}`;
            typeText(deviceLine, devText, () => {

                if (window.lastNumberOfConcepts) {
                    let conceptText = `Number of concepts per block: ${window.lastNumberOfConcepts}`;
                    typeText(conceptLine, conceptText, () => {
                        
                        if (codesStr) {
                            let codesText = `Activated Concepts for Each Block: ${codesStr}`;
                            setTimeout(() => {
                                typeText(codesLine, codesText, doneTyping);
                            }, 500);
                        } else {
                            codesLine.style.display = "none";
                            doneTyping();
                        }
                    });
                } else {
                    conceptLine.style.display = "none";
                    if (codesStr) {
                        let codesText = `Activated Concepts for Each Block: ${codesStr}`;
                        setTimeout(() => {
                            typeText(codesLine, codesText, doneTyping);
                        }, 500);
                    } else {
                        codesLine.style.display = "none";
                        doneTyping();
                    }
                }
            });
        } else {
            // If device is not available, skip the device line
            deviceLine.style.display = "none";

            if (window.lastNumberOfConcepts) {
                let conceptText = `Number of concepts per block: ${window.lastNumberOfConcepts}`;
                typeText(conceptLine, conceptText, () => {
                    if (codesStr) {
                        let codesText = `Activated Concepts for Each Block: ${codesStr}`;
                        setTimeout(() => {
                            typeText(codesLine, codesText, doneTyping);
                        }, 500);
                    } else {
                        codesLine.style.display = "none";
                        doneTyping();
                    }
                });
            } else {
                conceptLine.style.display = "none";
                if (codesStr) {
                    let codesText = `Activated Concepts for Each Block: ${codesStr}`;
                    setTimeout(() => {
                        typeText(codesLine, codesText, doneTyping);
                    }, 500);
                } else {
                    codesLine.style.display = "none";
                    doneTyping();
                }
            }
        }
    }

    // Toggle the feedback section for the currently visualized block
    window.toggleFeedbackSection = function() {
        let feedbackSection = document.getElementById("visualization-feedback-section");
        if (!feedbackSection) return;

        if (feedbackSection.style.display === "none" || feedbackSection.style.display === "") {
            feedbackSection.style.display = "block";
            window.isFeedbackOpen = true;
        } else {
            feedbackSection.style.display = "none";
            window.isFeedbackOpen = false;
            return;
        }

        let msgDiv = document.getElementById("feedback-not-available");
        let optsDiv = document.getElementById("feedback-options");
        // If no block has been visualized, show the "not available" message
        if (!window.isVisualizationPlot) {
            msgDiv.style.display = "block";
            optsDiv.style.display = "none";
        } else {
            msgDiv.style.display = "none";
            optsDiv.style.display = "block";
        }
    };

    // Handles form submission for all dynamic tasks: run_model, visualization, inspections, etc.
    function handleFormSubmission(event) {
        event.preventDefault();
        let form = event.target;
        let formData = new FormData(form);
        let endpoint = form.getAttribute("data-action");

        window.isVisualizationPlot = false;

        // If run_model is triggered, reset certain UI elements
        if (endpoint === "run_model") {
            clearPlotImages();
            hideModelInfo();
            let rmBtn = document.getElementById("runModelButton");
            if (rmBtn) highlightButton(rmBtn);

            // Close all accordion sections
            const allContents = document.querySelectorAll('.accordion-content');
            allContents.forEach(content => {
                content.style.display = 'none';
            });
			// Reset forms for visualization and inspections
            document.getElementById("visualization-form").reset();
            document.getElementById("implicit-inspection-form").reset();
            document.getElementById("comparative-inspection-form").reset();
            document.getElementById("interventional-inspection-form").reset();

            // Close feedback section
            let fbSec = document.getElementById("visualization-feedback-section");
            if (fbSec) {
                fbSec.style.display = "none";
            }
            window.isFeedbackOpen = false;

            // Disable UI to prevent repeated clicks
            disableAllButtons();
            disableAnalysisInputs();
            window.isAnalysisInProgress = true; 
            setGalleryDisabled(true);
            form.querySelector('button[type="submit"]').disabled = true;
            console.log("Run Model => cleared old plots, sections, feedback. (No spinner).");
        } else {
			// For other endpoints (visualization, inspection), show the spinner
            clearPlotImages();
            showSpinner();
            console.log(`Analysis => spinner shown. Endpoint: ${endpoint}.`);
        }

        // Make an AJAX call to the specified endpoint
        fetch(`/${endpoint}`, {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide spinner only for non-run_model tasks
            if (endpoint !== "run_model") {
                hideSpinner();
            } else {
                // If run_model succeeded, re-enable UI
                if (data.success) {
                    enableAllButtons();
                    enableAnalysisInputs();
                    window.isAnalysisInProgress = false; 
                    setGalleryDisabled(false);
                }
            }

            // Check success status from server response
            if (data.success) {
                console.log(`✅ Success: ${data.message}`);
                let alwaysTyped = (endpoint === "run_model");

                // Store number of concepts from run_model if present
                if (data.concepts_per_block_str) {
                    window.lastNumberOfConcepts = data.concepts_per_block_str;
                } else {
                    window.lastNumberOfConcepts = null;
                }

                // Update the device/codes lines in the model info box
                showModelInfo(data.device, data.codes_str, alwaysTyped, () => {
                    if (endpoint === "run_model") {
                        form.querySelector('button[type="submit"]').disabled = false;
                        window.modelHasRun = true;
                    }
                });

                // If there's a plot path, update the output section with the plot
                if (data.plot_path) {
                    updatePlotImage(data.plot_path);
                }
				
                // If the request is for visualization, mark the visualization state as active
                if (endpoint === "visualization") {
                    window.isVisualizationPlot = true;
                    let blockId = formData.get("block_id");
                    window.currentBlockId = blockId;

                    // If the feedback panel opened previously, show feedback options
                    if (window.isFeedbackOpen) {
                        let feedbackSection = document.getElementById("visualization-feedback-section");
                        if (feedbackSection) {
                            feedbackSection.style.display = "block";
                            let msgDiv = document.getElementById("feedback-not-available");
                            let optsDiv = document.getElementById("feedback-options");
                            msgDiv.style.display = "none";
                            optsDiv.style.display = "block";
                        }
                    }
                }
            } else {
                // On failure, show the error message in the output section
                console.error(`❌ Error: ${data.message}`);
                typePlotMessage(data.message);
                if (endpoint === "run_model") {
                    form.querySelector('button[type="submit"]').disabled = false;
                    window.isAnalysisInProgress = false; 
                    setGalleryDisabled(false);
                    enableAllButtons();
                    enableAnalysisInputs();
                } else {
                    hideSpinner();
                }
            }
        })
        .catch(error => {
            // Catch any network or server-side errors
            if (endpoint !== "run_model") {
                hideSpinner();
            } else {
                window.isAnalysisInProgress = false;
                setGalleryDisabled(false);
                enableAllButtons();
                enableAnalysisInputs();
            }
            console.error("❌ Error:", error);
            typePlotMessage(`Error: ${error}`);
            if (endpoint === "run_model") {
                form.querySelector('button[type="submit"]').disabled = false;
            }
        });
    }

    // Attach the above handler to all forms and store the final endpoint in a custom data attribute
    document.querySelectorAll("form").forEach(form => {
        let actionPath = form.action.split("/").pop();
        form.setAttribute("data-action", actionPath);
        form.addEventListener("submit", handleFormSubmission);
    });

    // Directly run block visualization with a simpler approach than the form
    window.runVisualization = function(blockId) {
        clearPlotImages();
        showSpinner();

        let imagePath = document.getElementById('selected-image-path-visualization');
        if (!imagePath) {
            console.error("No hidden input found for visualization form");
            hideSpinner();
            return;
        }

        let formData = new FormData();
        formData.append("image_path", imagePath.value);
        formData.append("block_id", blockId);

        fetch("/visualization", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            if (data.success) {
                console.log("✅ Visualization success");
                if (data.plot_path) {
                    updatePlotImage(data.plot_path);
                }
                showModelInfo(data.device, data.codes_str, false);
                window.isVisualizationPlot = true;
                window.currentBlockId = blockId;

                // If feedback panel is open, show feedback options for the current block
                if (window.isFeedbackOpen) {
                    let feedbackSection = document.getElementById("visualization-feedback-section");
                    if (feedbackSection) {
                        feedbackSection.style.display = "block";
                        let msgDiv = document.getElementById("feedback-not-available");
                        let optsDiv = document.getElementById("feedback-options");
                        msgDiv.style.display = "none";
                        optsDiv.style.display = "block";
                    }
                }
            } else {
                console.error(`❌ Error: ${data.message}`);
                typePlotMessage(data.message);
                window.isVisualizationPlot = false;
            }
        })
        .catch(err => {
            hideSpinner();
            console.error("❌ Visualization request failed:", err);
            typePlotMessage("Error: " + err);
            window.isVisualizationPlot = false;
        });
    };

    // Update the displayed image and reset relevant UI states
    function updateImage(newImage = null) {
        if (newImage) {
            index = images.indexOf(newImage);
            if (index === -1) index = 0;
        }
        let selectedImage = images[index];
        displayImage.src = `/images/${selectedImage}?t=${new Date().getTime()}`;
        selectedImageName.innerText = selectedImage;
        document.querySelectorAll("input[name='image_path']").forEach(input => {
            input.value = selectedImage;
        });
        console.log(`🔄 Updated Image: ${selectedImage}`);

        clearPlotImages();
        hideModelInfo();
        window.modelHasRun = false;
        window.isVisualizationPlot = false;
        window.currentBlockId = null;

        // Force close all accordion sections if open
        const allContents = document.querySelectorAll('.accordion-content');
        allContents.forEach(content => {
            content.style.display = 'none';
        });

        // Remove highlight from any previously pressed button
        if (window.lastPressedButton) {
            window.lastPressedButton.classList.remove("selected-button");
            window.lastPressedButton = null;
        }

        // Close feedback section if open
        let fbSec = document.getElementById("visualization-feedback-section");
        if (fbSec) {
            fbSec.style.display = "none";
        }
        window.isFeedbackOpen = false;
    }

    // Opens the gallery modal for selecting a new image
    window.openGallery = function() {
        // If either analysis is in progress OR typing device/codes => block
        if (window.isAnalysisInProgress || window.isTypingModelInfo) { 
            console.log("Gallery unavailable while analysis or typing is in progress.");
            return;
        }
        loadGalleryImages();
        let galleryModal = document.getElementById("galleryModal");
        let overlay = document.getElementById("overlay");
        galleryModal.style.display = "block";
        overlay.style.display = "block";
        setTimeout(() => {
            galleryModal.classList.add('open');
            overlay.classList.add('open');
        }, 10);
    };
	
    // Closes the gallery modal
    window.closeGallery = function() {
        let galleryModal = document.getElementById("galleryModal");
        let overlay = document.getElementById("overlay");
        galleryModal.classList.remove('open');
        overlay.classList.remove('open');
        setTimeout(() => {
            galleryModal.style.display = "none";
            overlay.style.display = "none";
        }, 300);
    };
	
    // Dynamically generate the gallery items using the images array
    function loadGalleryImages() {
        galleryContainer.innerHTML = "";
        images.forEach(imageName => {
            let galleryItem = document.createElement("div");
            galleryItem.className = "gallery-item";
            galleryItem.innerHTML = `
                <img src="/images/${imageName}" alt="Preview" loading="lazy" onclick="selectImage('${imageName}')">
                <span>${imageName}</span>
            `;
            galleryContainer.appendChild(galleryItem);
        });
        console.log("📸 Gallery images loaded.");
    }

    // When a user selects an image in the gallery, update the main display
    window.selectImage = function(imageName) {
        let idx = images.indexOf(imageName);
        if (idx !== -1) {
            index = idx;
            updateImage();
        }
        closeGallery();
    };

    // Go to the next image in the images array
    window.nextImage = function() {
        if (window.isAnalysisInProgress || window.isTypingModelInfo) {
            console.log("Gallery unavailable while analysis or typing is in progress.");
            return;
        }
        index = (index + 1) % images.length;
        updateImage();
    };

    // Go to the previous image in the images array
    window.prevImage = function() {
        if (window.isAnalysisInProgress || window.isTypingModelInfo) {
            console.log("Gallery unavailable while analysis or typing is in progress.");
            return;
        }
        index = (index - 1 + images.length) % images.length;
        updateImage();
    };

    // Toggle visibility of the accordion's content panels for various inspection modes
    window.toggleAccordion = function(sectionId) {
        // Ensure the user has run the model before any inspection operations
        if (!window.modelHasRun) {
            typePlotMessage("Please run the model to enable the Operation Mode.");
            return;
        }

        const allContents = document.querySelectorAll('.accordion-content');
        allContents.forEach(content => {
            if (content.id === sectionId) {
                if (content.style.display === 'none' || content.style.display === '') {
                    content.style.display = 'block';
                } else {
                    content.style.display = 'none';
                }
            } else {
                content.style.display = 'none';
            }
        });

        // If the Visualization section changed, reset that state
        if (sectionId !== "visualization-section") {
            window.isVisualizationPlot = false;
            window.currentBlockId = null;
            let feedbackSection = document.getElementById("visualization-feedback-section");
            if (feedbackSection) {
                feedbackSection.style.display = "none";
            }
            window.isFeedbackOpen = false;
        }
    };

    // Submit a predefined label (shape, color, size, etc.) as feedback for the visualized block
    window.submitFeedback = function(label) {
        if (!window.isVisualizationPlot || window.currentBlockId === null) {
            alert("Feedback is only available for a visualized block. Please visualize a block first.");
            return;
        }
        let formData = new FormData();
        formData.append("block_id", window.currentBlockId);
        formData.append("feedback_label", label);

        fetch("/save_feedback", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                typeFeedbackMessage(data.message, "#000");
            } else {
                typePlotMessage("Error: " + data.message);
            }
        })
        .catch(err => {
            typePlotMessage("Error: " + err);
        });
    };

    // Submit a custom text label as feedback for the visualized block
    window.submitFeedbackText = function() {
        if (!window.isVisualizationPlot || window.currentBlockId === null) {
            alert("Feedback is only available for a visualized block. Please visualize a block first.");
            return;
        }
        const textInput = document.getElementById("feedback-text-input");
        const label = textInput.value.trim();
        if(!label){
            typeFeedbackMessage("Please enter text for feedback to save!", "#B71C1C");
            return;
        }
        submitFeedback(label);
        textInput.value = "";
    };

    // Toggles the visibility of the "Actions" dropdown (links to GitHub, official NCB site, etc.)
    window.toggleActionsDropdown = function(event) {
        const dd = document.getElementById("actionsDropdown");
        dd.classList.toggle("show");
    };

    // Close the "Actions" dropdown if the user clicks outside of it
    window.addEventListener("click", function(e) {
        const dd = document.getElementById("actionsDropdown");
        const actionsBtn = document.getElementById("actionsButton");
        if (!actionsBtn.contains(e.target) && !dd.contains(e.target) && dd.classList.contains("show")) {
            dd.classList.remove("show");
        }
    });

    // --- Functions for the plot modal view ---
    // Opens a modal overlay with a larger view of the selected plot image
    window.openPlotModal = function(src) {
        let plotModal = document.getElementById("plotModal");
        let plotModalImage = document.getElementById("plotModalImage");
        plotModalImage.src = src;
        plotModal.classList.add('open');
    };

    // Closes the modal overlay for the plot image
    window.closePlotModal = function() {
        let plotModal = document.getElementById("plotModal");
        plotModal.classList.remove('open');
        setTimeout(() => {
            plotModal.style.display = "none";
            plotModal.style.removeProperty('display');
        }, 300);
    };
    // --- end plot modal functions ---

    // Initialize the displayed image, button highlighting, and feedback text input behavior
    updateImage();
    initSelectableButtons();

    // Enter button in the custom text feedback input to submit
    const fbInput = document.getElementById("feedback-text-input");
    if (fbInput) {
        fbInput.addEventListener("keyup", function(e) {
            if (e.key === "Enter") {
                submitFeedbackText();
            }
        });
    }
});
