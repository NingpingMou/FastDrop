//-This file contains the Introduction component that explains the project overview
import React from 'react';
import '../styles/Introduction.css';

const Introduction = () => {
  return (
    <section id="introduction" className="section">
      <h2 className="section-title">Introduction</h2>
      <div className="section-content">
        <div className="two-column">
          <div>
            <p>
              Deep Neural Networks (DNNs) have demonstrated remarkable performance in various tasks, 
              particularly in computer vision. However, they are vulnerable to adversarial attacks, 
              which are carefully crafted perturbations designed to mislead the model into making 
              incorrect predictions.
            </p>
            <p>
              <strong>FastDrop</strong> is a novel and efficient decision-based attack against black-box 
              models. Unlike existing adversarial attacks that rely on gradient estimation and additive 
              noise, FastDrop generates adversarial examples by dropping information in the frequency domain.
            </p>
            <div className="highlight-box">
              <p>
                <strong>Key Advantages:</strong>
              </p>
              <ul>
                <li>Requires only a few queries (10-20) to conduct an attack within 1 second</li>
                <li>Reduces the number of queries by 13-133× compared to state-of-the-art attacks</li>
                <li>Can escape detection by state-of-the-art black-box defenses</li>
                <li>Achieves 100% attack success rate on commercial vision APIs with just 10 queries on average</li>
              </ul>
            </div>
          </div>
          <div className="image-container">
            <img 
              src="/overview.png" 
              alt="FastDrop Overview" 
              className="img-responsive"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "https://via.placeholder.com/500x300?text=FastDrop+Overview";
              }}
            />
            <p className="image-caption">Overview of the FastDrop attack method</p>
          </div>
        </div>
        <div className="tags-container">
          <span className="tag primary">Adversarial Attacks</span>
          <span className="tag primary">Deep Neural Networks</span>
          <span className="tag secondary">Frequency Domain</span>
          <span className="tag secondary">Black-box Models</span>
          <span className="tag success">Decision-based</span>
          <span className="tag danger">Security</span>
        </div>
      </div>
    </section>
  );
};

export default Introduction;


import React from 'react';
import '../styles/Introduction.css';

const Introduction = () => {
  return (
    <section id="introduction" className="section">
      <h2 className="section-title">Introduction</h2>
      <div className="section-content">
        <div className="two-column">
          <div>
            <p>
              Deep Neural Networks (DNNs) have demonstrated remarkable performance in various tasks, 
              particularly in computer vision. However, they are vulnerable to adversarial attacks, 
              which are carefully crafted perturbations designed to mislead the model into making 
              incorrect predictions.
            </p>
            <p>
              <strong>FastDrop</strong> is a novel and efficient decision-based attack against black-box 
              models. Unlike existing adversarial attacks that rely on gradient estimation and additive 
              noise, FastDrop generates adversarial examples by dropping information in the frequency domain.
            </p>
            <div className="highlight-box">
              <p>
                <strong>Key Advantages:</strong>
              </p>
              <ul>
                <li>Requires only a few queries (10-20) to conduct an attack within 1 second</li>
                <li>Reduces the number of queries by 13-133× compared to state-of-the-art attacks</li>
                <li>Can escape detection by state-of-the-art black-box defenses</li>
                <li>Achieves 100% attack success rate on commercial vision APIs with just 10 queries on average</li>
              </ul>
            </div>
          </div>
          <div className="image-container">
            <img 
              src="/overview.png" 
              alt="FastDrop Overview" 
              className="img-responsive"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "https://via.placeholder.com/500x300?text=FastDrop+Overview";
              }}
            />
            <p className="image-caption">Overview of the FastDrop attack method</p>
          </div>
        </div>
        <div className="tags-container">
          <span className="tag primary">Adversarial Attacks</span>
          <span className="tag primary">Deep Neural Networks</span>
          <span className="tag secondary">Frequency Domain</span>
          <span className="tag secondary">Black-box Models</span>
          <span className="tag success">Decision-based</span>
          <span className="tag danger">Security</span>
        </div>
      </div>
    </section>
  );
};

export default Introduction;
