//-This file is the main component that composes the entire application
import React from 'react';
import Header from './components/Header';
import Navigation from './components/Navigation';
import Introduction from './components/Introduction';
import Methodology from './components/Methodology';
import Results from './components/Results';
import Comparison from './components/Comparison';
import Implementation from './components/Implementation';
import CodeExample from './components/CodeExample';
import References from './components/References';
import Footer from './components/Footer';
import './styles/App.css';

function App() {
  return (
    <div className="app">
      <Header />
      <Navigation />
      <main className="container">
        <Introduction />
        <Methodology />
        <Results />
        <Comparison />
        <Implementation />
        <CodeExample />
        <References />
      </main>
      <Footer />
    </div>
  );
}

export default App;


import React from 'react';
import Header from './components/Header';
import Navigation from './components/Navigation';
import Introduction from './components/Introduction';
import Methodology from './components/Methodology';
import Results from './components/Results';
import Comparison from './components/Comparison';
import Implementation from './components/Implementation';
import CodeExample from './components/CodeExample';
import References from './components/References';
import Footer from './components/Footer';
import './styles/App.css';

function App() {
  return (
    <div className="app">
      <Header />
      <Navigation />
      <main className="container">
        <Introduction />
        <Methodology />
        <Results />
        <Comparison />
        <Implementation />
        <CodeExample />
        <References />
      </main>
      <Footer />
    </div>
  );
}

export default App;
