//-This file contains the Implementation component that explains how to use FastDrop
import React from 'react';
import '../styles/Implementation.css';

const Implementation = () => {
  return (
    <section id="implementation" className="section">
      <h2 className="section-title">Implementation Details</h2>
      <div className="section-content">
        <p>
          FastDrop is implemented in Python using popular libraries such as NumPy, PyTorch, and 
          OpenCV. The implementation is designed to be modular and easy to use, allowing researchers 
          and practitioners to apply FastDrop to various models and datasets.
        </p>
        
        <div className="implementation-steps">
          <div className="implementation-step">
            <h3>1. Project Structure</h3>
            <p>
              The FastDrop project is organized into several modules:
            </p>
            <div className="code-structure">
              <pre>
                <code>
{`FastDrop/
├── attack-cifar10.py      # FastDrop implementation for CIFAR-10
├── attack-imagenet.py     # FastDrop implementation for ImageNet
├── models/                # Model definitions
│   ├── __init__.py
│   ├── resnet.py          # ResNet model architecture
│   └── resnet50.py        # ResNet50 model architecture
└── utils/                 # Utility functions
    ├── __init__.py
    └── util.py            # Helper functions`}
                </code>
              </pre>
            </div>
          </div>
          
          <div className="implementation-step">
            <h3>2. Dependencies</h3>
            <p>
              To use FastDrop, you'll need the following dependencies:
            </p>
            <ul className="dependencies-list">
              <li><span className="dependency-name">Python 3.6+</span></li>
              <li><span className="dependency-name">PyTorch 1.7+</span></li>
              <li><span className="dependency-name">NumPy</span></li>
              <li><span className="dependency-name">OpenCV</span></li>
              <li><span className="dependency-name">Matplotlib</span> (for visualization)</li>
              <li><span className="dependency-name">PIL</span> (Python Imaging Library)</li>
            </ul>
          </div>
          
          <div className="implementation-step">
            <h3>3. Usage</h3>
            <p>
              Using FastDrop is straightforward. Here's how to run the attack on CIFAR-10 and ImageNet:
            </p>
            <div className="usage-example">
              <div className="terminal">
                <div className="terminal-header">
                  <span className="terminal-button"></span>
                  <span className="terminal-button"></span>
                  <span className="terminal-button"></span>
                  <span className="terminal-title">Terminal</span>
                </div>
                <div className="terminal-content">
                  <pre>
                    <code>
{`# For CIFAR-10
python attack-cifar10.py

# For ImageNet
python attack-imagenet.py`}
                    </code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
          
          <div className="implementation-step">
            <h3>4. Customization</h3>
            <p>
              FastDrop can be customized for different models and datasets. Key parameters include:
            </p>
            <div className="parameters-table-container">
              <table className="parameters-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Description</th>
                    <th>Default Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>square_max_num</code></td>
                    <td>Maximum number of frequency components to consider</td>
                    <td>32</td>
                  </tr>
                  <tr>
                    <td><code>l2_norm_thres</code></td>
                    <td>L2 norm threshold for perturbation</td>
                    <td>5.0</td>
                  </tr>
                  <tr>
                    <td><code>block_size</code></td>
                    <td>Size of frequency blocks</td>
                    <td>16</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Performance Considerations</h4>
          <p>
            FastDrop is designed to be computationally efficient. On a standard GPU (e.g., NVIDIA 
            GTX 1080 Ti), it can generate adversarial examples within seconds. The attack is also 
            memory-efficient, making it suitable for deployment on resource-constrained environments.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Implementation;


import React from 'react';
import '../styles/Implementation.css';

const Implementation = () => {
  return (
    <section id="implementation" className="section">
      <h2 className="section-title">Implementation Details</h2>
      <div className="section-content">
        <p>
          FastDrop is implemented in Python using popular libraries such as NumPy, PyTorch, and 
          OpenCV. The implementation is designed to be modular and easy to use, allowing researchers 
          and practitioners to apply FastDrop to various models and datasets.
        </p>
        
        <div className="implementation-steps">
          <div className="implementation-step">
            <h3>1. Project Structure</h3>
            <p>
              The FastDrop project is organized into several modules:
            </p>
            <div className="code-structure">
              <pre>
                <code>
{`FastDrop/
├── attack-cifar10.py      # FastDrop implementation for CIFAR-10
├── attack-imagenet.py     # FastDrop implementation for ImageNet
├── models/                # Model definitions
│   ├── __init__.py
│   ├── resnet.py          # ResNet model architecture
│   └── resnet50.py        # ResNet50 model architecture
└── utils/                 # Utility functions
    ├── __init__.py
    └── util.py            # Helper functions`}
                </code>
              </pre>
            </div>
          </div>
          
          <div className="implementation-step">
            <h3>2. Dependencies</h3>
            <p>
              To use FastDrop, you'll need the following dependencies:
            </p>
            <ul className="dependencies-list">
              <li><span className="dependency-name">Python 3.6+</span></li>
              <li><span className="dependency-name">PyTorch 1.7+</span></li>
              <li><span className="dependency-name">NumPy</span></li>
              <li><span className="dependency-name">OpenCV</span></li>
              <li><span className="dependency-name">Matplotlib</span> (for visualization)</li>
              <li><span className="dependency-name">PIL</span> (Python Imaging Library)</li>
            </ul>
          </div>
          
          <div className="implementation-step">
            <h3>3. Usage</h3>
            <p>
              Using FastDrop is straightforward. Here's how to run the attack on CIFAR-10 and ImageNet:
            </p>
            <div className="usage-example">
              <div className="terminal">
                <div className="terminal-header">
                  <span className="terminal-button"></span>
                  <span className="terminal-button"></span>
                  <span className="terminal-button"></span>
                  <span className="terminal-title">Terminal</span>
                </div>
                <div className="terminal-content">
                  <pre>
                    <code>
{`# For CIFAR-10
python attack-cifar10.py

# For ImageNet
python attack-imagenet.py`}
                    </code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
          
          <div className="implementation-step">
            <h3>4. Customization</h3>
            <p>
              FastDrop can be customized for different models and datasets. Key parameters include:
            </p>
            <div className="parameters-table-container">
              <table className="parameters-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Description</th>
                    <th>Default Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>square_max_num</code></td>
                    <td>Maximum number of frequency components to consider</td>
                    <td>32</td>
                  </tr>
                  <tr>
                    <td><code>l2_norm_thres</code></td>
                    <td>L2 norm threshold for perturbation</td>
                    <td>5.0</td>
                  </tr>
                  <tr>
                    <td><code>block_size</code></td>
                    <td>Size of frequency blocks</td>
                    <td>16</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Performance Considerations</h4>
          <p>
            FastDrop is designed to be computationally efficient. On a standard GPU (e.g., NVIDIA 
            GTX 1080 Ti), it can generate adversarial examples within seconds. The attack is also 
            memory-efficient, making it suitable for deployment on resource-constrained environments.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Implementation;
