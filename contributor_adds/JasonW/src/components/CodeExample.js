//-This file contains the CodeExample component that shows code snippets of FastDrop
import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import '../styles/CodeExample.css';

const CodeExample = () => {
  const fastdropCode = `def fastdrop(net, img, block_size=16, file_path='', device=torch.device("cuda:0")):
    path1, path2 = os.path.split(file_path)
    save_path = path1 + '_adv/' + path2
    query_num = 0
    ori_img = img.copy()

    # Convert image to tensor and normalize
    img = trans1(img).unsqueeze(0)
    img = norm(img)
    img = img.to(device)
    query_num += 1
    with torch.no_grad():
        out = net(img)
        _, ori_label = torch.max(out, dim=1)

    # DFT - Convert to frequency domain
    freq = np.fft.fft2(ori_img, axes=(0, 1))
    freq_ori = freq.copy()
    freq_ori_m = np.abs(freq_ori)
    freq_abs = np.abs(freq)
    num_block = int(square_max_num/2)
    block_sum = np.zeros(num_block)
    
    # Calculate average magnitude for each block
    for i in range(num_block):
        block_sum[i] = square_avg(freq_abs, i)

    # Order blocks by importance
    block_sum_ind = np.argsort(block_sum)
    block_sum_ind_flag = np.zeros(num_block)

    # First stage: Iteratively drop frequency components
    freq_sec_stage = freq.copy()
    freq_sec_stage_m = np.abs(freq_sec_stage)  
    freq_sec_stage_p = np.angle(freq_sec_stage)  
    mag_start = 0
    
    for mag in range(1, num_block+1):
        for i in range(mag_start, mag):
            ind = block_sum_ind[i]
            freq_sec_stage_m = square_zero(freq_sec_stage_m, ind)
            freq_sec_stage = freq_sec_stage_m * np.e ** (1j * freq_sec_stage_p)

        # Convert back to spatial domain and check if attack is successful
        img_adv = np.abs(np.fft.ifft2(freq_sec_stage, axes=(0, 1)))
        img_adv = np.clip(img_adv, 0, 255)  
        img_adv = img_adv.astype('uint8')
        img_save = img_adv.copy()
        img_adv = trans1(img_adv).unsqueeze(0)
        img_adv = norm(img_adv)
        img_adv = img_adv.to(device)
        query_num += 1
        
        with torch.no_grad():
            out = net(img_adv)
            _, adv_label = torch.max(out, dim=1)

        mag_start = mag
        if ori_label != adv_label:
            # Attack successful
            l2_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2)
            if l2_norm.item() < l2_norm_thres:
                img_save = Image.fromarray(img_save)
                img_save.save(save_path)
                return
            break

    # Second stage: Optimize the adversarial example
    img_temp = img_save
    max_i = mag_start - 1
    block_sum_ind_flag[:max_i+1] = 1
    freq_m = freq_sec_stage_m
    freq_p = np.angle(freq)
    
    # Try to recover some frequency components while maintaining adversarial property
    optimize_block = 0
    for round in range(2):
        for i in range(max_i, -1, -1):
            if block_sum_ind_flag[i] == 1:
                ind = block_sum_ind[i]
                freq_m = square_recover(freq_m, freq_ori_m, ind)
                freq = freq_m * np.e ** (1j * freq_p)

                img_adv = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
                img_adv = np.clip(img_adv, 0, 255)
                img_adv = img_adv.astype('uint8')
                img_temp_2 = img_adv.copy()
                img_adv = trans1(img_adv).unsqueeze(0)
                img_adv = norm(img_adv)
                img_adv = img_adv.to(device)
                query_num += 1
                
                with torch.no_grad():
                    out = net(img_adv)
                    _, adv_label = torch.max(out, dim=1)

                if adv_label == ori_label:
                    # Recovering this component breaks the attack, zero it out again
                    freq_m = square_zero(freq_m, ind)
                    freq = freq_m * np.e ** (1j * freq_p)
                else:
                    # Successfully recovered this component while maintaining attack
                    img_temp = img_temp_2.copy()
                    optimize_block += 1
                    l2_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2)
                    block_sum_ind_flag[i] = 0

    # Save the final optimized adversarial example
    img_temp = Image.fromarray(img_temp)
    img_temp.save(save_path)`;

  const squareZeroCode = `def square_zero(freq:np.ndarray, index:int):
    freq_modified = freq.copy()
    freq_modified[index, index:square_max_num-index, :] = 0
    freq_modified[square_max_num-1-index, index:square_max_num-index, :] = 0
    freq_modified[index:square_max_num-index:, index, :] = 0
    freq_modified[index:square_max_num-index, square_max_num-1-index, :] = 0

    return freq_modified`;

  const squareRecoverCode = `def square_recover(freq_modified:np.ndarray, freq_ori:np.ndarray, index:int):
    freq_modified[index, index:square_max_num-index, :] = freq_ori[index, index:square_max_num-index, :]
    freq_modified[square_max_num-1-index, index:square_max_num-index, :] = freq_ori[square_max_num-1-index, index:square_max_num-index, :]
    freq_modified[index:square_max_num-index:, index, :] = freq_ori[index:square_max_num-index:, index, :]
    freq_modified[index:square_max_num-index, square_max_num-1-index, :] = freq_ori[index:square_max_num-index, square_max_num-1-index, :]

    return freq_modified`;

  return (
    <section id="code-example" className="section">
      <h2 className="section-title">Code Examples</h2>
      <div className="section-content">
        <p>
          Below are key code snippets from the FastDrop implementation. These examples illustrate 
          the core functionality of the attack method.
        </p>
        
        <div className="code-examples">
          <div className="code-example">
            <h3>Main FastDrop Function</h3>
            <p>
              This is the main function that implements the FastDrop attack. It takes an input image 
              and a target model, and returns an adversarial example.
            </p>
            <div className="code-container">
              <SyntaxHighlighter language="python" style={docco} showLineNumbers={true}>
                {fastdropCode}
              </SyntaxHighlighter>
            </div>
          </div>
          
          <div className="code-example-row">
            <div className="code-example-col">
              <h3>Square Zero Function</h3>
              <p>
                This function zeroes out specific frequency components in a square pattern.
              </p>
              <div className="code-container">
                <SyntaxHighlighter language="python" style={docco}>
                  {squareZeroCode}
                </SyntaxHighlighter>
              </div>
            </div>
            
            <div className="code-example-col">
              <h3>Square Recover Function</h3>
              <p>
                This function recovers specific frequency components from the original image.
              </p>
              <div className="code-container">
                <SyntaxHighlighter language="python" style={docco}>
                  {squareRecoverCode}
                </SyntaxHighlighter>
              </div>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Implementation Notes</h4>
          <ul>
            <li>
              The attack operates in two stages: first finding a successful adversarial example by 
              dropping frequency components, then optimizing it by recovering components that don't 
              affect the attack success.
            </li>
            <li>
              The <code>square_zero</code> and <code>square_recover</code> functions manipulate 
              specific patterns of frequency components, targeting the most important ones first.
            </li>
            <li>
              The implementation uses PyTorch for model inference and NumPy for frequency domain 
              operations, making it efficient and compatible with most deep learning workflows.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
};

export default CodeExample;


import React from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { docco } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import '../styles/CodeExample.css';

const CodeExample = () => {
  const fastdropCode = `def fastdrop(net, img, block_size=16, file_path='', device=torch.device("cuda:0")):
    path1, path2 = os.path.split(file_path)
    save_path = path1 + '_adv/' + path2
    query_num = 0
    ori_img = img.copy()

    # Convert image to tensor and normalize
    img = trans1(img).unsqueeze(0)
    img = norm(img)
    img = img.to(device)
    query_num += 1
    with torch.no_grad():
        out = net(img)
        _, ori_label = torch.max(out, dim=1)

    # DFT - Convert to frequency domain
    freq = np.fft.fft2(ori_img, axes=(0, 1))
    freq_ori = freq.copy()
    freq_ori_m = np.abs(freq_ori)
    freq_abs = np.abs(freq)
    num_block = int(square_max_num/2)
    block_sum = np.zeros(num_block)
    
    # Calculate average magnitude for each block
    for i in range(num_block):
        block_sum[i] = square_avg(freq_abs, i)

    # Order blocks by importance
    block_sum_ind = np.argsort(block_sum)
    block_sum_ind_flag = np.zeros(num_block)

    # First stage: Iteratively drop frequency components
    freq_sec_stage = freq.copy()
    freq_sec_stage_m = np.abs(freq_sec_stage)  
    freq_sec_stage_p = np.angle(freq_sec_stage)  
    mag_start = 0
    
    for mag in range(1, num_block+1):
        for i in range(mag_start, mag):
            ind = block_sum_ind[i]
            freq_sec_stage_m = square_zero(freq_sec_stage_m, ind)
            freq_sec_stage = freq_sec_stage_m * np.e ** (1j * freq_sec_stage_p)

        # Convert back to spatial domain and check if attack is successful
        img_adv = np.abs(np.fft.ifft2(freq_sec_stage, axes=(0, 1)))
        img_adv = np.clip(img_adv, 0, 255)  
        img_adv = img_adv.astype('uint8')
        img_save = img_adv.copy()
        img_adv = trans1(img_adv).unsqueeze(0)
        img_adv = norm(img_adv)
        img_adv = img_adv.to(device)
        query_num += 1
        
        with torch.no_grad():
            out = net(img_adv)
            _, adv_label = torch.max(out, dim=1)

        mag_start = mag
        if ori_label != adv_label:
            # Attack successful
            l2_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2)
            if l2_norm.item() < l2_norm_thres:
                img_save = Image.fromarray(img_save)
                img_save.save(save_path)
                return
            break

    # Second stage: Optimize the adversarial example
    img_temp = img_save
    max_i = mag_start - 1
    block_sum_ind_flag[:max_i+1] = 1
    freq_m = freq_sec_stage_m
    freq_p = np.angle(freq)
    
    # Try to recover some frequency components while maintaining adversarial property
    optimize_block = 0
    for round in range(2):
        for i in range(max_i, -1, -1):
            if block_sum_ind_flag[i] == 1:
                ind = block_sum_ind[i]
                freq_m = square_recover(freq_m, freq_ori_m, ind)
                freq = freq_m * np.e ** (1j * freq_p)

                img_adv = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
                img_adv = np.clip(img_adv, 0, 255)
                img_adv = img_adv.astype('uint8')
                img_temp_2 = img_adv.copy()
                img_adv = trans1(img_adv).unsqueeze(0)
                img_adv = norm(img_adv)
                img_adv = img_adv.to(device)
                query_num += 1
                
                with torch.no_grad():
                    out = net(img_adv)
                    _, adv_label = torch.max(out, dim=1)

                if adv_label == ori_label:
                    # Recovering this component breaks the attack, zero it out again
                    freq_m = square_zero(freq_m, ind)
                    freq = freq_m * np.e ** (1j * freq_p)
                else:
                    # Successfully recovered this component while maintaining attack
                    img_temp = img_temp_2.copy()
                    optimize_block += 1
                    l2_norm = torch.norm(un_norm(img.squeeze()) - un_norm(img_adv.squeeze()), p=2)
                    block_sum_ind_flag[i] = 0

    # Save the final optimized adversarial example
    img_temp = Image.fromarray(img_temp)
    img_temp.save(save_path)`;

  const squareZeroCode = `def square_zero(freq:np.ndarray, index:int):
    freq_modified = freq.copy()
    freq_modified[index, index:square_max_num-index, :] = 0
    freq_modified[square_max_num-1-index, index:square_max_num-index, :] = 0
    freq_modified[index:square_max_num-index:, index, :] = 0
    freq_modified[index:square_max_num-index, square_max_num-1-index, :] = 0

    return freq_modified`;

  const squareRecoverCode = `def square_recover(freq_modified:np.ndarray, freq_ori:np.ndarray, index:int):
    freq_modified[index, index:square_max_num-index, :] = freq_ori[index, index:square_max_num-index, :]
    freq_modified[square_max_num-1-index, index:square_max_num-index, :] = freq_ori[square_max_num-1-index, index:square_max_num-index, :]
    freq_modified[index:square_max_num-index:, index, :] = freq_ori[index:square_max_num-index:, index, :]
    freq_modified[index:square_max_num-index, square_max_num-1-index, :] = freq_ori[index:square_max_num-index, square_max_num-1-index, :]

    return freq_modified`;

  return (
    <section id="code-example" className="section">
      <h2 className="section-title">Code Examples</h2>
      <div className="section-content">
        <p>
          Below are key code snippets from the FastDrop implementation. These examples illustrate 
          the core functionality of the attack method.
        </p>
        
        <div className="code-examples">
          <div className="code-example">
            <h3>Main FastDrop Function</h3>
            <p>
              This is the main function that implements the FastDrop attack. It takes an input image 
              and a target model, and returns an adversarial example.
            </p>
            <div className="code-container">
              <SyntaxHighlighter language="python" style={docco} showLineNumbers={true}>
                {fastdropCode}
              </SyntaxHighlighter>
            </div>
          </div>
          
          <div className="code-example-row">
            <div className="code-example-col">
              <h3>Square Zero Function</h3>
              <p>
                This function zeroes out specific frequency components in a square pattern.
              </p>
              <div className="code-container">
                <SyntaxHighlighter language="python" style={docco}>
                  {squareZeroCode}
                </SyntaxHighlighter>
              </div>
            </div>
            
            <div className="code-example-col">
              <h3>Square Recover Function</h3>
              <p>
                This function recovers specific frequency components from the original image.
              </p>
              <div className="code-container">
                <SyntaxHighlighter language="python" style={docco}>
                  {squareRecoverCode}
                </SyntaxHighlighter>
              </div>
            </div>
          </div>
        </div>
        
        <div className="highlight-box">
          <h4>Implementation Notes</h4>
          <ul>
            <li>
              The attack operates in two stages: first finding a successful adversarial example by 
              dropping frequency components, then optimizing it by recovering components that don't 
              affect the attack success.
            </li>
            <li>
              The <code>square_zero</code> and <code>square_recover</code> functions manipulate 
              specific patterns of frequency components, targeting the most important ones first.
            </li>
            <li>
              The implementation uses PyTorch for model inference and NumPy for frequency domain 
              operations, making it efficient and compatible with most deep learning workflows.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
};

export default CodeExample;
