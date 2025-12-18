function predict() {
    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            school: 0,
            sex: 1,
            age: Number(age.value),
            studytime: Number(studytime.value),
            failures: Number(failures.value),
            schoolsup: Number(schoolsup.value),
            famsup: Number(famsup.value),
            absences: Number(absences.value)
        })
    })
    .then(res => res.json())
    .then(data => {
        result.innerText =
            `Risk: ${data.risk_level} | Probability: ${data.probability}`;
    });
}
