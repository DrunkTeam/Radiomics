document.getElementById('playButton').addEventListener('click', function () {
    const loading = document.getElementById('loading');
    loading.style.display = 'block';
    // Здесь будет код для отправки файлов на бекенд
});

document.getElementById('playButton').addEventListener('click', function () {
    const loading = document.getElementById('loading');
    loading.style.display = 'block';
    const progressBar = document.getElementById('progressBar');
    progressBar.style.display = 'block';
    let width = 0;
    const id = setInterval(frame, 10);
    function frame() {
        if (width >= 100) {
            clearInterval(id);
        } else {
            width++;
            progressBar.children[0].style.width = width + '%';
        }
    }
    // Здесь будет код для отправки файлов на бекенд
});

function switchLanguage() {
    var currentUrl = window.location.href;
    if (currentUrl.includes('index-ru.html')) {
        window.location.href = 'index.html';
    } else {
        window.location.href = 'index-ru.html';
    }
}

function checkFile(input, originalText, loadedText) {
    const file = input.files[0];
    const fileName = file.name;
    if (fileName.slice(fileName.lastIndexOf(".")) !== '.nii') {
        alert('Неверный формат файла. Пожалуйста, загрузите файл .nii');
        input.value = '';
        input.previousElementSibling.textContent = originalText;
    } else {
        input.previousElementSibling.textContent = loadedText;
    }
    checkBothFiles();
}

function checkBothFiles() {
    console.log("OK")
    const file1 = document.getElementById('file1');
    const file2 = document.getElementById('file2');
    const playButton = document.getElementById('playButton');
    if (file1.files.length > 0 && file2.files.length > 0) {
        playButton.disabled = false;
    } else {

    }
}