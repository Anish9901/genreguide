const dropArea = document.querySelector(".drop_box"),
  input = dropArea.querySelector("input"),
  button = dropArea.querySelector("button"),
  dragText = dropArea.querySelector("header");
let file;
var filename;

button.onclick = () => {
  input.click();
};

input.addEventListener("change", function (e) {
  var fileName = e.target.files[0].name;
  let filedata = `
    <form action="" method="post">
    <div class="form">
    <h4>${fileName}</h4>
    <button class="btn">Upload</button>
    </div>
    </form>`;
  dropArea.innerHTML = filedata;
});


/* //const input = document.querySelector('input');
const input = document.querySelector("input")
//input.style.opacity = 0;
console.log("hello")
console.log(input)
input.addEventListener('change', updateFileName);
function updateFileName() {
    const inputFile = input.files;
    console.log(inputFile[0].name)
} */