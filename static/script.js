document.getElementById("fileInput").addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const preview = document.getElementById("preview");
      preview.setAttribute("src", e.target.result);
      preview.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }
});
