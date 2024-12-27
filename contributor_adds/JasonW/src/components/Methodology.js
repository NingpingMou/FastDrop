//-This file contains the Methodology component that explains how FastDrop works
import React from 'react';
import '../styles/Methodology.css';

const Methodology = () => {
  return (
    <section id="methodology" className="section">
      <h2 className="section-title">Methodology</h2>
      <div className="section-content">
        <h3>How FastDrop Works</h3>
        <p>
          FastDrop operates by manipulating the frequency domain representation of images to create 
          adversarial examples. The key insight is that by selectively dropping information in the 
          frequency domain, we can create perturbations that are imperceptible to humans but can 
          effectively fool deep neural networks.
        </p>
        
        <div className="methodology-steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Discrete Fourier Transform (DFT)</h4>
              <p>
                The first step is to convert the input image from the spatial domain to the frequency 
                domain using the Discrete Fourier Transform (DFT). This transformation represents the 
                image as a combination of sinusoidal components with different frequencies.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Convert image to frequency domain
freq = np.fft.fft2(original_image, axes=(0, 1))`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Frequency Component Analysis</h4>
              <p>
                FastDrop analyzes the frequency components and identifies which ones contribute most 
                to the model's decision. This is done by calculating the average magnitude of frequency 
                components in different regions.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Calculate average magnitude for each block
for i in range(num_blocks):
    block_sum[i] = square_avg(freq_abs, i)`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Selective Information Dropping</h4>
              <p>
                Based on the analysis, FastDrop selectively zeroes out frequency components in order 
                of their importance. This process continues until the model's prediction changes, 
                indicating a successful adversarial example.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Zero out frequency components
freq_modified = square_zero(freq_modified, index)`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Inverse Transform & Optimization</h4>
              <p>
                The modified frequency representation is converted back to the spatial domain using 
                the Inverse Discrete Fourier Transform (IDFT). The resulting image is then optimized 
                to minimize the perturbation while maintaining the adversarial effect.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Convert back to spatial domain
img_adv = np.abs(np.fft.ifft2(freq_modified, axes=(0, 1)))
img_adv = np.clip(img_adv, 0, 255)  # Clip values to valid range`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Key Innovations</h4>
          <ul>
            <li>
              <strong>Frequency Domain Manipulation:</strong> Instead of adding noise in the spatial domain, 
              FastDrop works in the frequency domain, making it more efficient and harder to detect.
            </li>
            <li>
              <strong>Selective Information Dropping:</strong> By targeting specific frequency components, 
              FastDrop can create minimal perturbations that effectively fool the model.
            </li>
            <li>
              <strong>Decision-based Approach:</strong> FastDrop only requires access to the model's final 
              decision (top-1 label), making it applicable to real-world black-box scenarios.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
};

export default Methodology;


import React from 'react';
import '../styles/Methodology.css';

const Methodology = () => {
  return (
    <section id="methodology" className="section">
      <h2 className="section-title">Methodology</h2>
      <div className="section-content">
        <h3>How FastDrop Works</h3>
        <p>
          FastDrop operates by manipulating the frequency domain representation of images to create 
          adversarial examples. The key insight is that by selectively dropping information in the 
          frequency domain, we can create perturbations that are imperceptible to humans but can 
          effectively fool deep neural networks.
        </p>
        
        <div className="methodology-steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Discrete Fourier Transform (DFT)</h4>
              <p>
                The first step is to convert the input image from the spatial domain to the frequency 
                domain using the Discrete Fourier Transform (DFT). This transformation represents the 
                image as a combination of sinusoidal components with different frequencies.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Convert image to frequency domain
freq = np.fft.fft2(original_image, axes=(0, 1))`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Frequency Component Analysis</h4>
              <p>
                FastDrop analyzes the frequency components and identifies which ones contribute most 
                to the model's decision. This is done by calculating the average magnitude of frequency 
                components in different regions.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Calculate average magnitude for each block
for i in range(num_blocks):
    block_sum[i] = square_avg(freq_abs, i)`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Selective Information Dropping</h4>
              <p>
                Based on the analysis, FastDrop selectively zeroes out frequency components in order 
                of their importance. This process continues until the model's prediction changes, 
                indicating a successful adversarial example.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Zero out frequency components
freq_modified = square_zero(freq_modified, index)`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Inverse Transform & Optimization</h4>
              <p>
                The modified frequency representation is converted back to the spatial domain using 
                the Inverse Discrete Fourier Transform (IDFT). The resulting image is then optimized 
                to minimize the perturbation while maintaining the adversarial effect.
              </p>
              <div className="code-snippet">
                <pre>
                  <code>
                    {`# Convert back to spatial domain
img_adv = np.abs(np.fft.ifft2(freq_modified, axes=(0, 1)))
img_adv = np.clip(img_adv, 0, 255)  # Clip values to valid range`}
                  </code>
                </pre>
              </div>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Key Innovations</h4>
          <ul>
            <li>
              <strong>Frequency Domain Manipulation:</strong> Instead of adding noise in the spatial domain, 
              FastDrop works in the frequency domain, making it more efficient and harder to detect.
            </li>
            <li>
              <strong>Selective Information Dropping:</strong> By targeting specific frequency components, 
              FastDrop can create minimal perturbations that effectively fool the model.
            </li>
            <li>
              <strong>Decision-based Approach:</strong> FastDrop only requires access to the model's final 
              decision (top-1 label), making it applicable to real-world black-box scenarios.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
};

export default Methodology;
