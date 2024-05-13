const playButton = document.getElementById('playButton')
const loading = document.getElementById('loading')
const progressBar = document.getElementById('progressBar')
const file1 = document.getElementById('file1')
const file2 = document.getElementById('file2')

playButton.addEventListener('click', () => {
    loading.style.display = 'block'
    progressBar.style.display = 'block'

    let width = 0
    const frame = () => {
        if (width >= 99) {
            clearInterval(id)
        }

        width++
        progressBar.children[0].style.width = width + '%'
    }
    const id = setInterval(frame, 10)

    const formData = new FormData()

    formData.append('file1', file1.files[0])
    formData.append('file2', file2.files[0])

    fetch('/upload', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          console.log(data)
          //МОДИФИЦИРОВАТЬ ФРОНТ ТУТ
        })
        .catch(error => {
          console.error('Error uploading file:', error)
        })

        //ТУТ ПИСАТЬ НЕЛЬЗЯ
})

const switchLanguage = () => {
    window.location.href = window.location.href.includes('index-ru.html') ?
        'index.html' : 'index-ru.html'
}

const checkFile = (input, originalText, loadedText) => {
    const file = input.files[0]
    const fileName = file.name

    if (fileName.slice(fileName.lastIndexOf(".")) !== '.nii') {
        alert('Неверный формат файла. Пожалуйста, загрузите файл .nii')
        input.value = ''
        input.previousElementSibling.textContent = originalText
        return
    }

    input.previousElementSibling.textContent = loadedText

    if (file1.files.length > 0 && file2.files.length > 0) {
        playButton.disabled = false
    }
}