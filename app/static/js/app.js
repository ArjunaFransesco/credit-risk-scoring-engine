document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('risk-form');
    const btnAnalyze = document.getElementById('btn-analyze');
    const spinner = document.getElementById('form-spinner');
    const btnText = btnAnalyze.querySelector('.btn-text');

    // Result elements
    const scoreVal = document.getElementById('score-val');
    const scoreTier = document.getElementById('score-tier');
    const scoreCircle = document.querySelector('.score-circle');
    const badgeDecision = document.getElementById('badge-decision');
    const metricPd = document.getElementById('metric-pd');
    const meterPdFill = document.getElementById('meter-pd-fill');
    const metricDti = document.getElementById('metric-dti');
    const meterDtiFill = document.getElementById('meter-dti-fill');
    const metricGrade = document.getElementById('metric-grade');
    const factorList = document.getElementById('factor-list');

    // Presets
    const presets = {
        prime: {
            person_age: 38,
            person_income: 125000,
            loan_amnt: 15000,
            loan_int_rate: 6.5,
            person_emp_length: 9.0,
            cb_person_cred_hist_length: 12,
            person_home_ownership: 'MORTGAGE',
            loan_intent: 'HOMEIMPROVEMENT',
            loan_grade: 'A',
            cb_person_default_on_file: 'N'
        },
        median: {
            person_age: 29,
            person_income: 55000,
            loan_amnt: 12000,
            loan_int_rate: 11.5,
            person_emp_length: 3.5,
            cb_person_cred_hist_length: 5,
            person_home_ownership: 'RENT',
            loan_intent: 'PERSONAL',
            loan_grade: 'B',
            cb_person_default_on_file: 'N'
        },
        subprime: {
            person_age: 22,
            person_income: 24000,
            loan_amnt: 18000,
            loan_int_rate: 19.8,
            person_emp_length: 0.5,
            cb_person_cred_hist_length: 2,
            person_home_ownership: 'RENT',
            loan_intent: 'DEBTCONSOLIDATION',
            loan_grade: 'E',
            cb_person_default_on_file: 'Y'
        }
    };

    function applyPreset(data) {
        for (const [key, value] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) el.value = value;
        }
        triggerPrediction();
    }

    document.getElementById('preset-prime').addEventListener('click', () => applyPreset(presets.prime));
    document.getElementById('preset-median').addEventListener('click', () => applyPreset(presets.median));
    document.getElementById('preset-subprime').addEventListener('click', () => applyPreset(presets.subprime));

    async function triggerPrediction() {
        btnAnalyze.disabled = true;
        spinner.style.display = 'inline-block';
        btnText.textContent = 'Evaluating Risk Pipeline...';

        const formData = {
            person_age: parseInt(document.getElementById('person_age').value),
            person_income: parseFloat(document.getElementById('person_income').value),
            loan_amnt: parseFloat(document.getElementById('loan_amnt').value),
            loan_int_rate: parseFloat(document.getElementById('loan_int_rate').value),
            person_emp_length: parseFloat(document.getElementById('person_emp_length').value),
            cb_person_cred_hist_length: parseInt(document.getElementById('cb_person_cred_hist_length').value),
            person_home_ownership: document.getElementById('person_home_ownership').value,
            loan_intent: document.getElementById('loan_intent').value,
            loan_grade: document.getElementById('loan_grade').value,
            cb_person_default_on_file: document.getElementById('cb_person_default_on_file').value
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            if (result.status === 'success') {
                const data = result.data;

                // Animate score counter
                animateValue(scoreVal, parseInt(scoreVal.textContent) || 300, data.credit_score, 600);

                scoreTier.textContent = `${data.risk_category} (Grade ${data.risk_grade})`;
                scoreTier.style.color = data.color_code;
                scoreCircle.style.borderColor = data.color_code;
                scoreCircle.style.boxShadow = `0 0 25px ${data.color_code}40`;

                // Badge Decision
                badgeDecision.textContent = data.decision.replace('_', ' ');
                badgeDecision.className = `decision-badge ${data.decision.toLowerCase().replace('_', '-')}`;

                // Metrics
                const pdPercent = (data.probability_of_default * 100).toFixed(1);
                metricPd.textContent = `${pdPercent}%`;
                meterPdFill.style.width = `${Math.min(pdPercent, 100)}%`;
                meterPdFill.style.background = data.color_code;

                const dtiPercent = (data.debt_to_income_ratio * 100).toFixed(1);
                metricDti.textContent = `${dtiPercent}%`;
                meterDtiFill.style.width = `${Math.min(dtiPercent, 100)}%`;

                metricGrade.textContent = `Tier ${data.risk_grade}`;
                metricGrade.style.color = data.color_code;

                // Risk Factors
                factorList.innerHTML = '';
                if (data.risk_drivers && data.risk_drivers.length > 0) {
                    data.risk_drivers.forEach(driver => {
                        const li = document.createElement('li');
                        li.textContent = driver;
                        factorList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = 'No significant adverse risk drivers detected.';
                    factorList.appendChild(li);
                }
            }
        } catch (err) {
            console.error('Inference error:', err);
            alert('Failed to connect to ML scoring service.');
        } finally {
            btnAnalyze.disabled = false;
            spinner.style.display = 'none';
            btnText.textContent = 'Run Risk & Underwriting Assessment';
        }
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end;
            }
        };
        window.requestAnimationFrame(step);
    }

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        triggerPrediction();
    });

    // Run initial assessment on page load
    triggerPrediction();
});
