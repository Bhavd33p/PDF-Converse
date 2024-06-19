document.getElementById('process').addEventListener('click', () => {
    const pdfFiles = document.getElementById('pdf_files').files;
    const query = document.getElementById('query').value;

    if (pdfFiles.length === 0 || !query) {
        alert('Please upload PDF files and enter a query.');
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < pdfFiles.length; i++) {
        formData.append('pdf_files', pdfFiles[i]);
    }
    formData.append('query', query);

    fetch('http://localhost:5001/process_pdfs', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            alert(`Error: ${data.error}`);
            document.getElementById('result').textContent = `Error: ${data.error}`;
        } else {
            document.getElementById('result').textContent = `Answer: ${data.answer}`;
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        alert('An error occurred. Please try again later.');
        document.getElementById('result').textContent = `Error: ${error.message}`;
    });
});
