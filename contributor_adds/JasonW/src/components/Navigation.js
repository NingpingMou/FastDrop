//-This file contains the Navigation component for site navigation
import React, { useState, useEffect } from 'react';
import '../styles/Navigation.css';

const Navigation = () => {
  const [isSticky, setIsSticky] = useState(false);
  const [activeSection, setActiveSection] = useState('introduction');

  useEffect(() => {
    const handleScroll = () => {
      // Make navbar sticky when scrolled down
      setIsSticky(window.scrollY > 100);
      
      // Update active section based on scroll position
      const sections = document.querySelectorAll('section[id]');
      const scrollPosition = window.scrollY + 100;
      
      sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
          setActiveSection(sectionId);
        }
      });
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (sectionId) => {
    const section = document.getElementById(sectionId);
    if (section) {
      window.scrollTo({
        top: section.offsetTop - 70,
        behavior: 'smooth'
      });
    }
  };

  return (
    <nav className={`navigation ${isSticky ? 'sticky' : ''}`}>
      <div className="container nav-container">
        <ul className="nav-links">
          <li className={activeSection === 'introduction' ? 'active' : ''}>
            <button onClick={() => scrollToSection('introduction')}>Introduction</button>
          </li>
          <li className={activeSection === 'methodology' ? 'active' : ''}>
            <button onClick={() => scrollToSection('methodology')}>Methodology</button>
          </li>
          <li className={activeSection === 'results' ? 'active' : ''}>
            <button onClick={() => scrollToSection('results')}>Results</button>
          </li>
          <li className={activeSection === 'comparison' ? 'active' : ''}>
            <button onClick={() => scrollToSection('comparison')}>Comparison</button>
          </li>
          <li className={activeSection === 'implementation' ? 'active' : ''}>
            <button onClick={() => scrollToSection('implementation')}>Implementation</button>
          </li>
          <li className={activeSection === 'code-example' ? 'active' : ''}>
            <button onClick={() => scrollToSection('code-example')}>Code</button>
          </li>
          <li className={activeSection === 'references' ? 'active' : ''}>
            <button onClick={() => scrollToSection('references')}>References</button>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navigation;


import React, { useState, useEffect } from 'react';
import '../styles/Navigation.css';

const Navigation = () => {
  const [isSticky, setIsSticky] = useState(false);
  const [activeSection, setActiveSection] = useState('introduction');

  useEffect(() => {
    const handleScroll = () => {
      // Make navbar sticky when scrolled down
      setIsSticky(window.scrollY > 100);
      
      // Update active section based on scroll position
      const sections = document.querySelectorAll('section[id]');
      const scrollPosition = window.scrollY + 100;
      
      sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
          setActiveSection(sectionId);
        }
      });
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (sectionId) => {
    const section = document.getElementById(sectionId);
    if (section) {
      window.scrollTo({
        top: section.offsetTop - 70,
        behavior: 'smooth'
      });
    }
  };

  return (
    <nav className={`navigation ${isSticky ? 'sticky' : ''}`}>
      <div className="container nav-container">
        <ul className="nav-links">
          <li className={activeSection === 'introduction' ? 'active' : ''}>
            <button onClick={() => scrollToSection('introduction')}>Introduction</button>
          </li>
          <li className={activeSection === 'methodology' ? 'active' : ''}>
            <button onClick={() => scrollToSection('methodology')}>Methodology</button>
          </li>
          <li className={activeSection === 'results' ? 'active' : ''}>
            <button onClick={() => scrollToSection('results')}>Results</button>
          </li>
          <li className={activeSection === 'comparison' ? 'active' : ''}>
            <button onClick={() => scrollToSection('comparison')}>Comparison</button>
          </li>
          <li className={activeSection === 'implementation' ? 'active' : ''}>
            <button onClick={() => scrollToSection('implementation')}>Implementation</button>
          </li>
          <li className={activeSection === 'code-example' ? 'active' : ''}>
            <button onClick={() => scrollToSection('code-example')}>Code</button>
          </li>
          <li className={activeSection === 'references' ? 'active' : ''}>
            <button onClick={() => scrollToSection('references')}>References</button>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navigation;
