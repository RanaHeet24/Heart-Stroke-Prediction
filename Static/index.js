
const WEIGHTS = {
    "Weakness_One_Side": 15,
    "Speech_Trouble": 12,
    "Vision_Problem": 10,
    "Severe_Headache": 11,
    "Dizziness_Balance": 12,
    "High_BP": 8,
    "Diabetes": 6,
    "Heart_Disease": 6,
    "Smoke": 5,
    "Alcohol": 3,
    "Overweight": 4,
    "Sleep_Issue": 3,
    "Family_History": 5
};

function predictStroke() {
    let totalScore = 0;
    let contribList = [];

    for (let feature in WEIGHTS) {
        let radios = document.getElementsByName(feature);
        let val = 0;
        radios.forEach(r => { if(r.checked && r.value==="1") val=1; });
        let contrib = val ? WEIGHTS[feature] : 0;
        totalScore += contrib;
        contribList.push(`${feature}: +${contrib}%`);
    }

    // Simulate model base probability (like KNN output)
    let modelBase = Math.floor(Math.random()*30 + 20); // 20%-50%
    let finalRisk = Math.min(totalScore + modelBase, 99);

    let riskLevel = "LOW";
    if(finalRisk>65) riskLevel="HIGH";
    else if(finalRisk>35) riskLevel="MODERATE";

    document.getElementById("risk-value").innerText = finalRisk + "%";
    document.getElementById("risk-level-text").innerText = riskLevel + " RISK";

    // Show contribution list
    let ul = document.getElementById("contrib-list");
    ul.innerHTML = "";
    contribList.forEach(item => {
        let li = document.createElement("li");
        li.textContent = item;
        ul.appendChild(li);
    });

    document.getElementById("result").style.display = "block";
}
// ---------------------------
// Simulated "train_model.py"
// ---------------------------
function trainModel() {

    const WEIGHTS = {
        "Weakness_One_Side": 15,
        "Speech_Trouble": 12,
        "Vision_Problem": 10,
        "Severe_Headache": 11,
        "Dizziness_Balance": 12,
        "High_BP": 8,
        "Diabetes": 6,
        "Heart_Disease": 6,
        "Smoke": 5,
        "Alcohol": 3,
        "Overweight": 4,
        "Sleep_Issue": 3,
        "Family_History": 5
    };

    
    const model = {
        predictProba: (x) => {
           
            let score = 0;
            for(let i=0;i<x.length;i++){
                score += x[i]*Object.values(WEIGHTS)[i];
            }
            let prob = Math.min(score + Math.random()*30, 99)/100; // 0..1
            return [1-prob, prob];
        }
    };

    return { WEIGHTS, model };
}

// ---------------------------
// Simulated "app.py"
// ---------------------------
function analyzeRisk(inputs) {
    const { WEIGHTS, model } = trainModel();

   
    const featureOrder = Object.keys(WEIGHTS);
    const x_vec = featureOrder.map(f => inputs[f] || 0);

    
    const prob = model.predictProba(x_vec)[1]; // 0..1

    let riskLevel = "LOW";
    if(prob>0.65) riskLevel="HIGH";
    else if(prob>0.35) riskLevel="MODERATE";

    
    let contribList = [];
    featureOrder.forEach((f,i)=>{
        const contrib = x_vec[i] ? WEIGHTS[f] : 0;
        contribList.push(`${f}: +${contrib}%`);
    });

    return { riskPercent: (prob*100).toFixed(1), riskLevel, contribList };
}


const userInput = {
    "Weakness_One_Side":1,
    "Speech_Trouble":0,
    "Vision_Problem":1,
    "Severe_Headache":0,
    "Dizziness_Balance":1,
    "High_BP":1,
    "Diabetes":0,
    "Heart_Disease":0,
    "Smoke":1,
    "Alcohol":0,
    "Overweight":1,
    "Sleep_Issue":0,
    "Family_History":0
};

const result = analyzeRisk(userInput);
console.log(`Predicted Stroke Risk: ${result.riskPercent}% -> ${result.riskLevel}`);
console.log("Feature contributions:", result.contribList);
