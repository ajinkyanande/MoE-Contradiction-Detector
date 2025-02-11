async function classifyText() {
    const paragraph = document.getElementById("inputText").value;
    const response = await fetch("http://localhost:8000/classify/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paragraph })
    });
    const data = await response.json();
    
    displayResults(data);
}

function displayResults(data) {
    const resultDiv = document.getElementById("results");
    resultDiv.innerHTML = "";

    const colors = ["red", "blue", "green", "purple", "orange"];
    let colorMap = {};
    let colorIndex = 0;

    data.entail_groups.forEach(group => {
        const color = colors[colorIndex % colors.length];
        group.forEach(idx => { colorMap[idx] = color; });
        colorIndex++;
    });

    data.sentences.forEach((sentence, index) => {
        const span = document.createElement("span");
        span.innerText = sentence + " ";
        if (index in colorMap) {
            span.style.backgroundColor = colorMap[index];
        }
        resultDiv.appendChild(span);
    });

    const contradictionsDiv = document.createElement("div");
    contradictionsDiv.innerHTML = "<h3>Contradictions:</h3>";
    data.contradictions.forEach(([i, j]) => {
        const p = document.createElement("p");
        p.innerHTML = `<strong>${data.sentences[i]}</strong> contradicts <strong>${data.sentences[j]}</strong>`;
        contradictionsDiv.appendChild(p);
    });

    resultDiv.appendChild(contradictionsDiv);
}
