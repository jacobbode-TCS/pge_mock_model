const preview = document.getElementById('image-preview');
const fileInput = document.getElementById('image-file');
const dropZone = document.getElementById('drop-zone');
const fileName = document.getElementById('file-name');
const previewContainer = document.getElementById('preview-container');
const clearButton = document.getElementById('clear-image');
let selectedFile = null;
const form = document.getElementById('workflow-form');
const requestResult = document.getElementById('request-result');
const chosenAgent = document.getElementById('chosen-agent');
const outputHeading = document.getElementById('output-heading');
const output = document.getElementById('output');
const sourcesContainer = document.getElementById('sources');
const sourcesContent = document.getElementById('sources-content');

function formatOutputValue(value) {
    if (typeof value === 'object' && value !== null) {
        return JSON.stringify(value, null, 2);
    }
    return value || '';
}

function updateOutputSection(data) {
    const response = data.error ? data.error : data.response;
    output.innerHTML = marked.parse(formatOutputValue(response));
}

function updateSourcesSection(data) {
    const sources = data.sources || null;

    if (!sources || (Array.isArray(sources) && sources.length === 0)) {
        sourcesContainer.style.display = 'none';
        return;
    }

    sourcesContainer.style.display = 'block';
    sourcesContent.innerHTML = '';
    sources.forEach(source => {
        const details = document.createElement('details');
        details.className = 'source-item';
        const summary = document.createElement('summary');
        const link = document.createElement('a');
        link.href = source.url;
        link.textContent = source.title || source.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        
        // Prevent expanding when clicking the link itself
        link.addEventListener('click', e => e.stopPropagation());
        summary.appendChild(link);
        const content = document.createElement('div');
        content.className = 'source-content';
        content.textContent = source.content || '';
        details.appendChild(summary);
        details.appendChild(content);
        sourcesContent.appendChild(details);
    });
}

function showPreview(file) {

    const reader = new FileReader();

    reader.onload = function (e) {

        preview.src = e.target.result;

        dropZone.style.display = 'none';
        previewContainer.style.display = 'block';

        fileName.textContent = file.name;
    };

    reader.readAsDataURL(file);
}

function clearImage() {

    selectedFile = null;

    fileInput.value = '';

    preview.src = '';

    previewContainer.style.display = 'none';
    dropZone.style.display = 'block';

    fileName.textContent = '';
}

clearButton.addEventListener('click', clearImage);
dropZone.addEventListener('click', () => {

    if (selectedFile) {
        return;
    }

    fileInput.click();
});

fileInput.addEventListener('change', (event) => {
    selectedFile = event.target.files[0];

    if (selectedFile) {
        fileName.textContent = selectedFile.name;
        showPreview(selectedFile);
    }
});

dropZone.addEventListener('dragover', (event) => {

    if (selectedFile) {
        return;
    }

    event.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (event) => {

    if (selectedFile) {
        return;
    }

    event.preventDefault();

    dropZone.classList.remove('dragover');

    selectedFile = event.dataTransfer.files[0];

    if (selectedFile) {
        showPreview(selectedFile);
    }
});

preview.addEventListener('click', () => {
    fileInput.click();
});

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData();

    formData.append(
        'request',
        document.getElementById('request').value
    );

    if (selectedFile) {
        formData.append('image', selectedFile);
    }

    requestResult.textContent = 'Running...';
    chosenAgent.textContent = 'Running...';
    output.textContent = 'Running...';
    const res = await fetch('/workflow', {
        method: 'POST',
        body: formData
    });

    const data = await res.json();
    requestResult.textContent = data.request || '';
    chosenAgent.textContent = data.chosen_agent || '';
    updateOutputSection(data);
    updateSourcesSection(data);
});