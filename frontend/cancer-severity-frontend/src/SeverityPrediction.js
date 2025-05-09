import React, { useState } from 'react';
import axios from 'axios';
import './SeverityPrediction.css';

const SeverityPrediction = () => {
  const [isActive, setIsActive] = useState(true);
  const [response, setResponse] = useState("");
  const [data, setData] = useState({
    Genetic_Risk: 0,
    Air_Pollution: 0,
    Alcohol_Use: 0,
    Smoking: 0,
    Obesity_Level: 0,
    Treatment_Cost_USD: 0,
    Survival_Years: 0,
    Age: 0,
    Gender: 'Male',
    Country_Region: 'USA',
    Cancer_Type: 'Lung',
    Cancer_Stage: 'Stage I',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setData({ ...data, [name]: isNaN(value) ? value : Number(value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await axios.post('http://127.0.0.1:9000/predict', data, {
        headers: { 'Content-Type': 'application/json' },
      });

      console.log(res.data);
      setResponse(res.data.predictions[0]); // Adjust if response format is different
      setIsActive(false);
    } catch (error) {
      console.error("Error during prediction:", error);
      alert("Error connecting to backend");
    }
  };

  return (
    <div>
      {isActive ? (
        <div className="container">
          <h2 className="title">Predict Cancer Severity</h2>
          <form className="form" onSubmit={handleSubmit}>
            {[
              { label: 'Genetic Risk', name: 'Genetic_Risk' },
              { label: 'Air Pollution', name: 'Air_Pollution' },
              { label: 'Alcohol Use', name: 'Alcohol_Use' },
              { label: 'Smoking', name: 'Smoking' },
              { label: 'Obesity Level', name: 'Obesity_Level' },
              { label: 'Treatment Cost (USD)', name: 'Treatment_Cost_USD' },
              { label: 'Survival Years', name: 'Survival_Years' },
              { label: 'Age', name: 'Age' },
            ].map((field) => (
              <div className="form-group" key={field.name}>
                <label>{field.label}</label>
                <input
                  type="number"
                  name={field.name}
                  value={data[field.name]}
                  onChange={handleChange}
                />
              </div>
            ))}

            <div className="form-group">
              <label>Gender</label>
              <select name="Gender" value={data.Gender} onChange={handleChange}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className="form-group">
              <label>Country / Region</label>
              <select name="Country_Region" value={data.Country_Region} onChange={handleChange}>
                <option value="USA">USA</option>
                <option value="UK">UK</option>
                <option value="France">France</option>
                <option value="Germany">Germany</option>
                <option value="Canada">Canada</option>
                <option value="Italy">Italy</option>
                <option value="Spain">Spain</option>
                <option value="Australia">Australia</option>
                <option value="India">India</option>
              </select>
            </div>

            <div className="form-group">
              <label>Cancer Type</label>
              <select name="Cancer_Type" value={data.Cancer_Type} onChange={handleChange}>
                <option value="Lung">Lung</option>
                <option value="Breast">Breast</option>
                <option value="Colon">Colon</option>
                <option value="Prostate">Prostate</option>
                <option value="Skin">Skin</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className="form-group">
              <label>Cancer Stage</label>
              <select name="Cancer_Stage" value={data.Cancer_Stage} onChange={handleChange}>
                <option value="Stage I">Stage I</option>
                <option value="Stage II">Stage II</option>
                <option value="Stage III">Stage III</option>
                <option value="Stage IV">Stage IV</option>
              </select>
            </div>

            <button type="submit" className="submit-btn">Predict</button>
          </form>
        </div>
      ) : (
        <div className="container result">
          <h2>Prediction Result</h2>
          <h3>Severity Class: {response}</h3>
          <button onClick={() => setIsActive(true)} className="submit-btn">
            Predict Again
          </button>
        </div>
      )}
    </div>
  );
};

export default SeverityPrediction;
