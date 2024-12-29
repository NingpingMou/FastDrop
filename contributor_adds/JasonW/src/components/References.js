//-This file contains the References component that lists citations and related work
import React from 'react';
import '../styles/References.css';

const References = () => {
  return (
    <section id="references" className="section">
      <h2 className="section-title">References</h2>
      <div className="section-content">
        <div className="references-container">
          <div className="citation-box">
            <h3>Citation</h3>
            <div className="citation-content">
              <pre>
                <code>
{`@inproceedings{DBLP:conf/ijcai/MouZWGG22,
  author    = {Ningping Mou and
              Baolin Zheng and
              Qian Wang and
              Yunjie Ge and
              Binqing Guo},
  title     = {A Few Seconds Can Change Everything: Fast Decision-based Attacks against
              DNNs},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
              Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
              2022},
  year      = {2022}
}`}
                </code>
              </pre>
            </div>
          </div>
          
          <h3>Related Work</h3>
          <div className="references-list">
            <div className="reference-item">
              <div className="reference-number">1</div>
              <div className="reference-content">
                <p className="reference-title">
                  Boundary Attack: Decision-based Adversarial Attack in the Absence of Confidence Scores
                </p>
                <p className="reference-authors">
                  Wieland Brendel, Jonas Rauber, Matthias Bethge
                </p>
                <p className="reference-publication">
                  International Conference on Learning Representations (ICLR), 2018
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">2</div>
              <div className="reference-content">
                <p className="reference-title">
                  HopSkipJumpAttack: A Query-Efficient Decision-Based Attack
                </p>
                <p className="reference-authors">
                  Jianbo Chen, Michael I. Jordan, Martin J. Wainwright
                </p>
                <p className="reference-publication">
                  IEEE Symposium on Security and Privacy (SP), 2020
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">3</div>
              <div className="reference-content">
                <p className="reference-title">
                  Sign-OPT: A Query-Efficient Hard-label Adversarial Attack
                </p>
                <p className="reference-authors">
                  Minhao Cheng, Simranjit Singh, Patrick H. Chen, Pin-Yu Chen, Sijia Liu, Cho-Jui Hsieh
                </p>
                <p className="reference-publication">
                  International Conference on Learning Representations (ICLR), 2020
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">4</div>
              <div className="reference-content">
                <p className="reference-title">
                  Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search
                </p>
                <p className="reference-authors">
                  Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, Matthias Hein
                </p>
                <p className="reference-publication">
                  European Conference on Computer Vision (ECCV), 2020
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">5</div>
              <div className="reference-content">
                <p className="reference-title">
                  Towards Evaluating the Robustness of Neural Networks
                </p>
                <p className="reference-authors">
                  Nicholas Carlini, David Wagner
                </p>
                <p className="reference-publication">
                  IEEE Symposium on Security and Privacy (SP), 2017
                </p>
              </div>
            </div>
          </div>
          
          <h3>Resources</h3>
          <div className="resources-list">
            <a href="https://github.com/NingpingMou/FastDrop" target="_blank" rel="noopener noreferrer" className="resource-link">
              <div className="resource-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                </svg>
              </div>
              <div className="resource-info">
                <span className="resource-title">GitHub Repository</span>
                <span className="resource-url">github.com/NingpingMou/FastDrop</span>
              </div>
            </a>
            
            <a href="https://www.ijcai.org/proceedings/2022/464" target="_blank" rel="noopener noreferrer" className="resource-link">
              <div className="resource-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
              </div>
              <div className="resource-info">
                <span className="resource-title">IJCAI 2022 Paper</span>
                <span className="resource-url">ijcai.org/proceedings/2022/464</span>
              </div>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default References;


import React from 'react';
import '../styles/References.css';

const References = () => {
  return (
    <section id="references" className="section">
      <h2 className="section-title">References</h2>
      <div className="section-content">
        <div className="references-container">
          <div className="citation-box">
            <h3>Citation</h3>
            <div className="citation-content">
              <pre>
                <code>
{`@inproceedings{DBLP:conf/ijcai/MouZWGG22,
  author    = {Ningping Mou and
              Baolin Zheng and
              Qian Wang and
              Yunjie Ge and
              Binqing Guo},
  title     = {A Few Seconds Can Change Everything: Fast Decision-based Attacks against
              DNNs},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
              Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
              2022},
  year      = {2022}
}`}
                </code>
              </pre>
            </div>
          </div>
          
          <h3>Related Work</h3>
          <div className="references-list">
            <div className="reference-item">
              <div className="reference-number">1</div>
              <div className="reference-content">
                <p className="reference-title">
                  Boundary Attack: Decision-based Adversarial Attack in the Absence of Confidence Scores
                </p>
                <p className="reference-authors">
                  Wieland Brendel, Jonas Rauber, Matthias Bethge
                </p>
                <p className="reference-publication">
                  International Conference on Learning Representations (ICLR), 2018
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">2</div>
              <div className="reference-content">
                <p className="reference-title">
                  HopSkipJumpAttack: A Query-Efficient Decision-Based Attack
                </p>
                <p className="reference-authors">
                  Jianbo Chen, Michael I. Jordan, Martin J. Wainwright
                </p>
                <p className="reference-publication">
                  IEEE Symposium on Security and Privacy (SP), 2020
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">3</div>
              <div className="reference-content">
                <p className="reference-title">
                  Sign-OPT: A Query-Efficient Hard-label Adversarial Attack
                </p>
                <p className="reference-authors">
                  Minhao Cheng, Simranjit Singh, Patrick H. Chen, Pin-Yu Chen, Sijia Liu, Cho-Jui Hsieh
                </p>
                <p className="reference-publication">
                  International Conference on Learning Representations (ICLR), 2020
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">4</div>
              <div className="reference-content">
                <p className="reference-title">
                  Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search
                </p>
                <p className="reference-authors">
                  Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, Matthias Hein
                </p>
                <p className="reference-publication">
                  European Conference on Computer Vision (ECCV), 2020
                </p>
              </div>
            </div>
            
            <div className="reference-item">
              <div className="reference-number">5</div>
              <div className="reference-content">
                <p className="reference-title">
                  Towards Evaluating the Robustness of Neural Networks
                </p>
                <p className="reference-authors">
                  Nicholas Carlini, David Wagner
                </p>
                <p className="reference-publication">
                  IEEE Symposium on Security and Privacy (SP), 2017
                </p>
              </div>
            </div>
          </div>
          
          <h3>Resources</h3>
          <div className="resources-list">
            <a href="https://github.com/NingpingMou/FastDrop" target="_blank" rel="noopener noreferrer" className="resource-link">
              <div className="resource-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                </svg>
              </div>
              <div className="resource-info">
                <span className="resource-title">GitHub Repository</span>
                <span className="resource-url">github.com/NingpingMou/FastDrop</span>
              </div>
            </a>
            
            <a href="https://www.ijcai.org/proceedings/2022/464" target="_blank" rel="noopener noreferrer" className="resource-link">
              <div className="resource-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
              </div>
              <div className="resource-info">
                <span className="resource-title">IJCAI 2022 Paper</span>
                <span className="resource-url">ijcai.org/proceedings/2022/464</span>
              </div>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default References;
