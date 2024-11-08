// Show spinner on form submission
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("prediction-form");
    const spinnerOverlay = document.getElementById("spinner-overlay");

    if (form) {
        form.addEventListener("submit", function() {
            // Show the spinner overlay and disable the submit button to prevent multiple submissions
            spinnerOverlay.style.visibility = "visible";
        });
    }
});
// scripts.js
document.getElementById("image_input").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith("image/")) {
        const imgPreview = document.createElement("img");
        imgPreview.src = URL.createObjectURL(file);
        imgPreview.style.maxWidth = "100%";
        
        const form = document.getElementById("prediction-form");
        form.insertBefore(imgPreview, document.getElementById("predict-button"));
    }
});