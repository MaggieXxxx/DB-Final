
document.addEventListener('DOMContentLoaded', function() {
    console.log('loaded')
    const ssnBtn = document.getElementById('ssnBtn');
    ssnBtn.addEventListener('click', function(e) {
        e.preventDefault();
        const ssn = document.getElementById('ssn').value;
        console.log(ssn)
        fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ssn: ssn }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('data', data);
            // Load main contents of the page here
            const ssnForm = document.getElementById('ssnForm');
            if (!data.error) {
                
                const navBar = document.getElementById('navbar-container');
                navBar.style.display = 'block';
                // Remove the SSN form here
                ssnForm.style.display = 'none';

                // display welcome message
                displayWelcome(data)
            } else {
                // Display error message
                const error = document.getElementById('ssnError');
                error.textContent = 'Error: Invalid SSN'
                error.style.color = 'red';
            }
            
        })
        .catch(error => console.log('Error:', error));
    });
})


function displayWelcome(data) {
    const welcome = document.getElementById('welcome');
    welcome.style.display = 'block';
    welcome.textContent = 'Welcome, ' + data.FirstName + '!\n' + 'What would you like to do today?';
    const subText = document.createElement('p');
    subText.textContent = 'Please select an option using the navigation menu above. Generate a new quote, view your current policy, or file a claim!';
    welcome.appendChild(subText);
}