document.getElementById("fileInput").addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById("preview").setAttribute("src", e.target.result);
            document.getElementById("preview").style.display = "block";
        };
        reader.readAsDataURL(file);
    }
});
