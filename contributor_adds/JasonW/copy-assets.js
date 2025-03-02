//-This file contains a script to copy assets from the main project to the public directory
const fs = require('fs');
const path = require('path');

// Source and destination paths
const sourceDir = '../../';
const destDir = './public';

// Files to copy
const filesToCopy = [
  'overview.png'
];

// Create destination directory if it doesn't exist
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

// Copy each file
filesToCopy.forEach(file => {
  const sourcePath = path.join(sourceDir, file);
  const destPath = path.join(destDir, file);
  
  try {
    // Read the source file
    const data = fs.readFileSync(sourcePath);
    
    // Write to the destination
    fs.writeFileSync(destPath, data);
    
    console.log(`Successfully copied ${file} to ${destPath}`);
  } catch (err) {
    console.error(`Error copying ${file}: ${err.message}`);
  }
});

console.log('Asset copying complete!');


const fs = require('fs');
const path = require('path');

// Source and destination paths
const sourceDir = '../../';
const destDir = './public';

// Files to copy
const filesToCopy = [
  'overview.png'
];

// Create destination directory if it doesn't exist
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

// Copy each file
filesToCopy.forEach(file => {
  const sourcePath = path.join(sourceDir, file);
  const destPath = path.join(destDir, file);
  
  try {
    // Read the source file
    const data = fs.readFileSync(sourcePath);
    
    // Write to the destination
    fs.writeFileSync(destPath, data);
    
    console.log(`Successfully copied ${file} to ${destPath}`);
  } catch (err) {
    console.error(`Error copying ${file}: ${err.message}`);
  }
});

console.log('Asset copying complete!');
