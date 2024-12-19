//-This file contains the Header component that displays the project title and logo
import React from 'react';
import '../styles/Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="container header-content">
        <div className="logo">
          <span className="logo-text">FastDrop</span>
        </div>
        <h1 className="title">A Few Seconds Can Change Everything</h1>
        <p className="subtitle">Fast Decision-based Attacks against Deep Neural Networks</p>
      </div>
    </header>
  );
};

export default Header;


import React from 'react';
import '../styles/Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="container header-content">
        <div className="logo">
          <span className="logo-text">FastDrop</span>
        </div>
        <h1 className="title">A Few Seconds Can Change Everything</h1>
        <p className="subtitle">Fast Decision-based Attacks against Deep Neural Networks</p>
      </div>
    </header>
  );
};

export default Header;
