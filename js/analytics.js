// analytics.js - Client-side tracking script for custom analytics

// Function to collect and send data to the API
async function trackEvent(actionName, additionalData = {}) {
  try {
    // Collect client-side data matching your columns
    const data = {
      action_name: actionName,
      is_first_action: !sessionStorage.getItem('session_started'), // True if first action in session
      online: navigator.onLine,
      country: '', // Can be enriched server-side via IP in track.js
      unique_ip_addresses: '', // Captured server-side
      screen_resolution: `${screen.width}x${screen.height}`,
      ip_address: '', // Captured server-side
      browser: navigator.userAgent, // Full user-agent (parse if needed)
      event_duration: additionalData.duration || 0,
      operating_system: navigator.platform,
      duration_of_visit: Math.floor(performance.now() / 1000), // Time since page load
      city: '', // Enriched server-side
      language: navigator.language,
      domain: window.location.hostname,
      landing_page_path: window.location.pathname
    };

    // Mark session as started to track 'is_first_action'
    if (data.is_first_action) {
      sessionStorage.setItem('session_started', 'true');
    }

    // Send to your Vercel API endpoint
    const response = await fetch('https://[website-name]-github-io-analytic.vercel.app/api/track', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      console.error('Tracking failed:', response.statusText);
    }
  } catch (error) {
    console.error('Error in tracking:', error);
  }
}

// Automatically track page view on load
window.addEventListener('load', () => trackEvent('page_view'));

// Optional: Add custom event tracking (e.g., for buttons or links)
// Example: document.getElementById('myButton').addEventListener('click', () => trackEvent('button_click', { duration: 5 }));