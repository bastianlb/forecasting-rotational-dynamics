// Main JavaScript file
document.addEventListener('DOMContentLoaded', function() {
  console.log('Website loaded successfully');
  
  // Simple image loading fallback for PDF images
  const images = document.querySelectorAll('img[src$=".pdf"]');
  images.forEach(img => {
    img.onerror = function() {
      // If PDF fails to load, show placeholder
      this.style.background = '#f5f5f5';
      this.style.border = '2px dashed #ccc';
      this.style.display = 'block';
      this.style.height = '300px';
      this.style.width = '100%';
      this.alt = 'PDF Figure - ' + this.alt;
    };
  });
});