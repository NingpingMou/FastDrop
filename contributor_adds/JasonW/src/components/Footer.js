//-This file contains the Footer component that displays at the bottom of the page
import React from 'react';
import '../styles/Footer.css';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="container footer-content">
        <div className="footer-section">
          <h3>FastDrop Project</h3>
          <p>
            A Few Seconds Can Change Everything: Fast Decision-based Attacks against DNNs
          </p>
        </div>
        
        <div className="footer-section">
          <h3>Quick Links</h3>
          <ul className="footer-links">
            <li><a href="#introduction">Introduction</a></li>
            <li><a href="#methodology">Methodology</a></li>
            <li><a href="#results">Results</a></li>
            <li><a href="#code-example">Code Examples</a></li>
          </ul>
        </div>
        
        <div className="footer-section">
          <h3>Resources</h3>
          <ul className="footer-links">
            <li>
              <a 
                href="https://github.com/NingpingMou/FastDrop" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                GitHub Repository
              </a>
            </li>
            <li>
              <a 
                href="https://www.ijcai.org/proceedings/2022/464" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                IJCAI 2022 Paper
              </a>
            </li>
          </ul>
        </div>
      </div>
      
      <div className="footer-bottom">
        <div className="container">
          <p>&copy; {currentYear} FastDrop Project. All rights reserved.</p>
          <p className="footer-disclaimer">
            This website is for educational and research purposes only.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;


import React from 'react';
import '../styles/Footer.css';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="container footer-content">
        <div className="footer-section">
          <h3>FastDrop Project</h3>
          <p>
            A Few Seconds Can Change Everything: Fast Decision-based Attacks against DNNs
          </p>
        </div>
        
        <div className="footer-section">
          <h3>Quick Links</h3>
          <ul className="footer-links">
            <li><a href="#introduction">Introduction</a></li>
            <li><a href="#methodology">Methodology</a></li>
            <li><a href="#results">Results</a></li>
            <li><a href="#code-example">Code Examples</a></li>
          </ul>
        </div>
        
        <div className="footer-section">
          <h3>Resources</h3>
          <ul className="footer-links">
            <li>
              <a 
                href="https://github.com/NingpingMou/FastDrop" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                GitHub Repository
              </a>
            </li>
            <li>
              <a 
                href="https://www.ijcai.org/proceedings/2022/464" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                IJCAI 2022 Paper
              </a>
            </li>
          </ul>
        </div>
      </div>
      
      <div className="footer-bottom">
        <div className="container">
          <p>&copy; {currentYear} FastDrop Project. All rights reserved.</p>
          <p className="footer-disclaimer">
            This website is for educational and research purposes only.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
