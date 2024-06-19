document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const categoryDiv = document.getElementById('category');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(form);
        const data = {};

        formData.forEach((value, key) => {
            data[key] = parseFloat(value) || value;
        });

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = `Error: ${data.error}`;
                categoryDiv.innerHTML = '';
            } else {
                resultDiv.innerHTML = `Prediction: ${data.prediction.toFixed(2)}`;
                categoryDiv.innerHTML = `Category: ${data.prediction_category}`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.innerHTML = 'Error al realizar la predicción';
            categoryDiv.innerHTML = '';
        });
    });
});
