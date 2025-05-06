import React, { useState } from 'react';
import axios from 'axios';
import './SeverityPrediction.css'; // Make sure you create this CSS file

const SeverityPrediction = () => {
  const [data, setData] = useState({
    geneticRisk: 0,
    airPollution: 0,
    alcoholUse: 0,
    smoking: 0,
    obesityLevel: 0,
    treatmentCost: 0,
    survivalYears: 0,
    age: 0,
    gender: 'Male',
    countryRegion: 'USA',
    cancerType: 'Lung',
    cancerStage: 'Stage 1',
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setData({ ...data, [name]: value });
  };

  const handleSubmit = async (e) => {
   alert(" ok bbb");
  };

  return (
    <div className="container">
      <h2 className="title">Predict Cancer Severity</h2>
      <form className="form" onSubmit={handleSubmit}>
        {[
          { label: 'Genetic Risk', name: 'geneticRisk' },
          { label: 'Air Pollution', name: 'airPollution' },
          { label: 'Alcohol Use', name: 'alcoholUse' },
          { label: 'Smoking', name: 'smoking' },
          { label: 'Obesity Level', name: 'obesityLevel' },
          { label: 'Treatment Cost (USD)', name: 'treatmentCost' },
          { label: 'Survival Years', name: 'survivalYears' },
          { label: 'Age', name: 'age' },
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
          <select name="gender" value={data.gender} onChange={handleChange}>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div className="form-group">
          <label>Country / Region</label>
          <select name="countryRegion" value={data.countryRegion} onChange={handleChange}>
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
          <select name="cancerType" value={data.cancerType} onChange={handleChange}>
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
          <select name="cancerStage" value={data.cancerStage} onChange={handleChange}>
            <option value="Stage 1">Stage 1</option>
            <option value="Stage 2">Stage 2</option>
            <option value="Stage 3">Stage 3</option>
            <option value="Stage 4">Stage 4</option>
          </select>
        </div>

        <button type="submit" className="submit-btn">Predict</button>
      </form>

      {prediction && (
        <div className="result">
          <h3>Prediction: Severity Class {prediction.severityClass}</h3>
        </div>
      )}
    </div>
  );
};

export default SeverityPrediction;
