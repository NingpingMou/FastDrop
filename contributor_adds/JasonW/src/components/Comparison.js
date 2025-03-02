//-This file contains the Comparison component that compares FastDrop with other methods
import React from 'react';
import '../styles/Comparison.css';

const Comparison = () => {
  return (
    <section id="comparison" className="section">
      <h2 className="section-title">Comparison with Other Methods</h2>
      <div className="section-content">
        <p>
          FastDrop represents a significant advancement over existing decision-based adversarial 
          attack methods. Here, we compare FastDrop with other state-of-the-art approaches across 
          various dimensions.
        </p>
        
        <div className="comparison-table-container">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Approach</th>
                <th>Query Efficiency</th>
                <th>Perturbation Size</th>
                <th>Defense Evasion</th>
              </tr>
            </thead>
            <tbody>
              <tr className="highlighted-row">
                <td><strong>FastDrop</strong></td>
                <td>Frequency domain information dropping</td>
                <td>Very High<br/>(10-20 queries)</td>
                <td>Low</td>
                <td>Excellent</td>
              </tr>
              <tr>
                <td>HSJA</td>
                <td>Boundary optimization</td>
                <td>Low<br/>(1000+ queries)</td>
                <td>Medium</td>
                <td>Good</td>
              </tr>
              <tr>
                <td>SignOPT</td>
                <td>Gradient estimation</td>
                <td>Medium<br/>(200+ queries)</td>
                <td>Medium</td>
                <td>Moderate</td>
              </tr>
              <tr>
                <td>Square Attack</td>
                <td>Random search</td>
                <td>Medium<br/>(200+ queries)</td>
                <td>High</td>
                <td>Moderate</td>
              </tr>
              <tr>
                <td>Boundary Attack</td>
                <td>Random walk</td>
                <td>Very Low<br/>(5000+ queries)</td>
                <td>Medium</td>
                <td>Poor</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <h3>Key Differences</h3>
        <div className="comparison-cards">
          <div className="comparison-card">
            <h4>Frequency Domain vs. Spatial Domain</h4>
            <p>
              Unlike most existing methods that operate in the spatial domain by directly 
              modifying pixel values, FastDrop works in the frequency domain. This approach 
              allows for more efficient and effective perturbations that are less detectable 
              by defense mechanisms.
            </p>
          </div>
          
          <div className="comparison-card">
            <h4>Information Dropping vs. Additive Noise</h4>
            <p>
              Traditional methods add carefully crafted noise to images. In contrast, FastDrop 
              selectively removes information from the frequency domain. This fundamental 
              difference contributes to FastDrop's superior query efficiency and ability to 
              bypass defenses.
            </p>
          </div>
          
          <div className="comparison-card">
            <h4>Optimization Strategy</h4>
            <p>
              FastDrop employs a novel optimization strategy that prioritizes frequency 
              components based on their importance to the model's decision. This targeted 
              approach is more efficient than the random search or gradient estimation 
              strategies used by other methods.
            </p>
          </div>
          
          <div className="comparison-card">
            <h4>Real-world Applicability</h4>
            <p>
              With its extremely low query requirements, FastDrop is particularly threatening 
              in real-world scenarios where API calls are limited or costly. Other methods 
              requiring hundreds or thousands of queries are less practical for attacking 
              commercial systems.
            </p>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Why FastDrop Outperforms Others</h4>
          <p>
            The superior performance of FastDrop can be attributed to its innovative approach 
            of manipulating the frequency domain rather than the spatial domain. By targeting 
            the fundamental information structure of images, FastDrop can achieve successful 
            attacks with minimal queries while generating perturbations that are difficult for 
            defense mechanisms to detect.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Comparison;


import React from 'react';
import '../styles/Comparison.css';

const Comparison = () => {
  return (
    <section id="comparison" className="section">
      <h2 className="section-title">Comparison with Other Methods</h2>
      <div className="section-content">
        <p>
          FastDrop represents a significant advancement over existing decision-based adversarial 
          attack methods. Here, we compare FastDrop with other state-of-the-art approaches across 
          various dimensions.
        </p>
        
        <div className="comparison-table-container">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Approach</th>
                <th>Query Efficiency</th>
                <th>Perturbation Size</th>
                <th>Defense Evasion</th>
              </tr>
            </thead>
            <tbody>
              <tr className="highlighted-row">
                <td><strong>FastDrop</strong></td>
                <td>Frequency domain information dropping</td>
                <td>Very High<br/>(10-20 queries)</td>
                <td>Low</td>
                <td>Excellent</td>
              </tr>
              <tr>
                <td>HSJA</td>
                <td>Boundary optimization</td>
                <td>Low<br/>(1000+ queries)</td>
                <td>Medium</td>
                <td>Good</td>
              </tr>
              <tr>
                <td>SignOPT</td>
                <td>Gradient estimation</td>
                <td>Medium<br/>(200+ queries)</td>
                <td>Medium</td>
                <td>Moderate</td>
              </tr>
              <tr>
                <td>Square Attack</td>
                <td>Random search</td>
                <td>Medium<br/>(200+ queries)</td>
                <td>High</td>
                <td>Moderate</td>
              </tr>
              <tr>
                <td>Boundary Attack</td>
                <td>Random walk</td>
                <td>Very Low<br/>(5000+ queries)</td>
                <td>Medium</td>
                <td>Poor</td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <h3>Key Differences</h3>
        <div className="comparison-cards">
          <div className="comparison-card">
            <h4>Frequency Domain vs. Spatial Domain</h4>
            <p>
              Unlike most existing methods that operate in the spatial domain by directly 
              modifying pixel values, FastDrop works in the frequency domain. This approach 
              allows for more efficient and effective perturbations that are less detectable 
              by defense mechanisms.
            </p>
          </div>
          
          <div className="comparison-card">
            <h4>Information Dropping vs. Additive Noise</h4>
            <p>
              Traditional methods add carefully crafted noise to images. In contrast, FastDrop 
              selectively removes information from the frequency domain. This fundamental 
              difference contributes to FastDrop's superior query efficiency and ability to 
              bypass defenses.
            </p>
          </div>
          
          <div className="comparison-card">
            <h4>Optimization Strategy</h4>
            <p>
              FastDrop employs a novel optimization strategy that prioritizes frequency 
              components based on their importance to the model's decision. This targeted 
              approach is more efficient than the random search or gradient estimation 
              strategies used by other methods.
            </p>
          </div>
          
          <div className="comparison-card">
            <h4>Real-world Applicability</h4>
            <p>
              With its extremely low query requirements, FastDrop is particularly threatening 
              in real-world scenarios where API calls are limited or costly. Other methods 
              requiring hundreds or thousands of queries are less practical for attacking 
              commercial systems.
            </p>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Why FastDrop Outperforms Others</h4>
          <p>
            The superior performance of FastDrop can be attributed to its innovative approach 
            of manipulating the frequency domain rather than the spatial domain. By targeting 
            the fundamental information structure of images, FastDrop can achieve successful 
            attacks with minimal queries while generating perturbations that are difficult for 
            defense mechanisms to detect.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Comparison;
