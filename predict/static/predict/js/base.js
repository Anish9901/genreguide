const dropArea = document.querySelector(".drop_box"),
  input = dropArea.querySelector("input"),
  button = dropArea.querySelector("label"),
  dragText = dropArea.querySelector("header");
let file;
var filename;

/* button.onclick = () => {
  input.click();
}; */

input.addEventListener("change", function (e) {
  var fileName = e.target.files[0].name;
  const upload = document.getElementById('b1')
  const header = document.getElementById('header')
  const support = document.getElementById('support')
  let uploadhtml= `<button id="b2" type="submit" class="btn">Upload</button>`;
  let headerhtml= `<h3>${fileName}</h3>`;
  let supporthtml=``;
  upload.innerHTML = uploadhtml;
  header.innerHTML = headerhtml;
  support.innerHTML = supporthtml;
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