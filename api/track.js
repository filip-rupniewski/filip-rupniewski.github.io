// /api/track.js (CORRECTED AND FINAL VERSION)

const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

module.exports = async (req, res) => {
  // --- START OF CORS HANDLING ---
  // Set CORS headers to allow requests from your GitHub Pages site
  res.setHeader('Access-Control-Allow-Origin', 'https://filip-rupniewski.github.io'); // IMPORTANT: REPLACE WITH YOUR GITHUB PAGES URL
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle the browser's preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(204).end(); // Respond with 204 No Content
    return;
  }
  // --- END OF CORS HANDLING ---


  // Handle the actual POST request from your analytics script
  if (req.method === 'POST') {
    try {
      const data = req.body;
      
      // Capture server-side info
      data.ip_address = req.headers['x-forwarded-for']?.split(',').shift() || req.socket.remoteAddress;
      data.timestamp = new Date().toISOString();

      // Your existing logic to insert data into Supabase
      const { error } = await supabase.from('analytics').insert([data]);

      if (error) {
        console.error('Supabase error:', error);
        return res.status(500).json({ error: error.message });
      }

      return res.status(200).json({ success: true });

    } catch (e) {
      console.error('Server error:', e);
      return res.status(500).json({ error: 'Internal Server Error' });
    }
  }

  // If the method is not POST or OPTIONS, return 405 Method Not Allowed
  res.setHeader('Allow', ['POST', 'OPTIONS']);
  res.status(405).json({ error: `Method ${req.method} Not Allowed` });
};