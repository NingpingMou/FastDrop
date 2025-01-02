//-This file contains the Results component that displays the experimental results
import React from 'react';
import '../styles/Results.css';

const Results = () => {
  return (
    <section id="results" className="section">
      <h2 className="section-title">Experimental Results</h2>
      <div className="section-content">
        <p>
          We conducted extensive experiments on three datasets (ImageNet, CIFAR-10, and CIFAR-100) 
          to evaluate the performance of FastDrop compared to state-of-the-art decision-based attacks.
        </p>
        
        <div className="results-grid">
          <div className="result-card">
            <h3>Query Efficiency</h3>
            <div className="result-content">
              <p>
                FastDrop significantly reduces the number of queries required to generate successful 
                adversarial examples:
              </p>
              <ul>
                <li>Only 10-20 queries needed to conduct an attack within 1 second</li>
                <li>13-133× fewer queries compared to state-of-the-art attacks under the same level of perturbations</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="bar" style={{ height: '30%' }}>
                    <span className="bar-label">FastDrop</span>
                    <span className="bar-value">15</span>
                  </div>
                  <div className="bar" style={{ height: '80%' }}>
                    <span className="bar-label">SignOPT</span>
                    <span className="bar-value">200</span>
                  </div>
                  <div className="bar" style={{ height: '100%' }}>
                    <span className="bar-label">HSJA</span>
                    <span className="bar-value">250</span>
                  </div>
                </div>
                <div className="chart-caption">Average number of queries required for successful attacks</div>
              </div>
            </div>
          </div>
          
          <div className="result-card">
            <h3>Attack Success Rate</h3>
            <div className="result-content">
              <p>
                FastDrop achieves high attack success rates across different models and datasets:
              </p>
              <ul>
                <li>100% success rate on commercial vision APIs (Baidu, Tencent) with just 10 queries on average</li>
                <li>High success rate even against models with state-of-the-art defenses</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="pie-chart">
                    <div className="pie-slice" style={{ transform: 'rotate(0deg)', backgroundColor: '#3498db' }}></div>
                    <div className="pie-slice" style={{ transform: 'rotate(360deg)', backgroundColor: 'transparent' }}></div>
                    <div className="pie-label">100%</div>
                  </div>
                </div>
                <div className="chart-caption">Attack success rate on commercial APIs</div>
              </div>
            </div>
          </div>
          
          <div className="result-card">
            <h3>Perturbation Size</h3>
            <div className="result-content">
              <p>
                FastDrop generates adversarial examples with minimal perturbations:
              </p>
              <ul>
                <li>Lower L2 norm compared to other decision-based attacks</li>
                <li>Perturbations are less perceptible to human observers</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="line-chart">
                    <div className="line-point" style={{ bottom: '20%', left: '10%' }}></div>
                    <div className="line-point" style={{ bottom: '40%', left: '30%' }}></div>
                    <div className="line-point" style={{ bottom: '60%', left: '50%' }}></div>
                    <div className="line-point" style={{ bottom: '70%', left: '70%' }}></div>
                    <div className="line-point" style={{ bottom: '80%', left: '90%' }}></div>
                  </div>
                </div>
                <div className="chart-caption">L2 norm vs. number of queries</div>
              </div>
            </div>
          </div>
          
          <div className="result-card">
            <h3>Defense Evasion</h3>
            <div className="result-content">
              <p>
                FastDrop can effectively evade state-of-the-art black-box defenses:
              </p>
              <ul>
                <li>Successfully bypasses input transformation defenses</li>
                <li>Effective against adversarial training defenses</li>
                <li>Works well against randomization-based defenses</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="stacked-bar">
                    <div className="stacked-segment" style={{ height: '70%', backgroundColor: '#3498db' }}></div>
                    <div className="stacked-segment" style={{ height: '30%', backgroundColor: '#e74c3c' }}></div>
                  </div>
                </div>
                <div className="chart-caption">Success rate against defended models</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Key Findings</h4>
          <p>
            Our experiments demonstrate that FastDrop represents a significant advancement in 
            decision-based adversarial attacks, offering superior performance in terms of query 
            efficiency, attack success rate, and defense evasion. The results highlight the 
            potential real-world threats posed by such efficient attacks and underscore the 
            need for more robust defense mechanisms.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Results;


import React from 'react';
import '../styles/Results.css';

const Results = () => {
  return (
    <section id="results" className="section">
      <h2 className="section-title">Experimental Results</h2>
      <div className="section-content">
        <p>
          We conducted extensive experiments on three datasets (ImageNet, CIFAR-10, and CIFAR-100) 
          to evaluate the performance of FastDrop compared to state-of-the-art decision-based attacks.
        </p>
        
        <div className="results-grid">
          <div className="result-card">
            <h3>Query Efficiency</h3>
            <div className="result-content">
              <p>
                FastDrop significantly reduces the number of queries required to generate successful 
                adversarial examples:
              </p>
              <ul>
                <li>Only 10-20 queries needed to conduct an attack within 1 second</li>
                <li>13-133× fewer queries compared to state-of-the-art attacks under the same level of perturbations</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="bar" style={{ height: '30%' }}>
                    <span className="bar-label">FastDrop</span>
                    <span className="bar-value">15</span>
                  </div>
                  <div className="bar" style={{ height: '80%' }}>
                    <span className="bar-label">SignOPT</span>
                    <span className="bar-value">200</span>
                  </div>
                  <div className="bar" style={{ height: '100%' }}>
                    <span className="bar-label">HSJA</span>
                    <span className="bar-value">250</span>
                  </div>
                </div>
                <div className="chart-caption">Average number of queries required for successful attacks</div>
              </div>
            </div>
          </div>
          
          <div className="result-card">
            <h3>Attack Success Rate</h3>
            <div className="result-content">
              <p>
                FastDrop achieves high attack success rates across different models and datasets:
              </p>
              <ul>
                <li>100% success rate on commercial vision APIs (Baidu, Tencent) with just 10 queries on average</li>
                <li>High success rate even against models with state-of-the-art defenses</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="pie-chart">
                    <div className="pie-slice" style={{ transform: 'rotate(0deg)', backgroundColor: '#3498db' }}></div>
                    <div className="pie-slice" style={{ transform: 'rotate(360deg)', backgroundColor: 'transparent' }}></div>
                    <div className="pie-label">100%</div>
                  </div>
                </div>
                <div className="chart-caption">Attack success rate on commercial APIs</div>
              </div>
            </div>
          </div>
          
          <div className="result-card">
            <h3>Perturbation Size</h3>
            <div className="result-content">
              <p>
                FastDrop generates adversarial examples with minimal perturbations:
              </p>
              <ul>
                <li>Lower L2 norm compared to other decision-based attacks</li>
                <li>Perturbations are less perceptible to human observers</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="line-chart">
                    <div className="line-point" style={{ bottom: '20%', left: '10%' }}></div>
                    <div className="line-point" style={{ bottom: '40%', left: '30%' }}></div>
                    <div className="line-point" style={{ bottom: '60%', left: '50%' }}></div>
                    <div className="line-point" style={{ bottom: '70%', left: '70%' }}></div>
                    <div className="line-point" style={{ bottom: '80%', left: '90%' }}></div>
                  </div>
                </div>
                <div className="chart-caption">L2 norm vs. number of queries</div>
              </div>
            </div>
          </div>
          
          <div className="result-card">
            <h3>Defense Evasion</h3>
            <div className="result-content">
              <p>
                FastDrop can effectively evade state-of-the-art black-box defenses:
              </p>
              <ul>
                <li>Successfully bypasses input transformation defenses</li>
                <li>Effective against adversarial training defenses</li>
                <li>Works well against randomization-based defenses</li>
              </ul>
              <div className="result-chart">
                <div className="chart-placeholder">
                  <div className="stacked-bar">
                    <div className="stacked-segment" style={{ height: '70%', backgroundColor: '#3498db' }}></div>
                    <div className="stacked-segment" style={{ height: '30%', backgroundColor: '#e74c3c' }}></div>
                  </div>
                </div>
                <div className="chart-caption">Success rate against defended models</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Key Findings</h4>
          <p>
            Our experiments demonstrate that FastDrop represents a significant advancement in 
            decision-based adversarial attacks, offering superior performance in terms of query 
            efficiency, attack success rate, and defense evasion. The results highlight the 
            potential real-world threats posed by such efficient attacks and underscore the 
            need for more robust defense mechanisms.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Results;
