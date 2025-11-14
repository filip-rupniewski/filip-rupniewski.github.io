// /api/track.js (CORRECTED AND FINAL VERSION)

const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

module.exports = async (req, res) => {
  // CORS Headers (keep these as they are)
  res.setHeader('Access-Control-Allow-Origin', 'https://filip-rupniewski.github.io'); // IMPORTANT: REPLACE WITH YOUR GITHUB PAGES URL
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

  if (req.method === 'POST') {
    try {
      const data = req.body;
      
      // Capture server-side info
      data.ip_address = req.headers['x-forwarded-for']?.split(',').shift() || req.socket.remoteAddress;
      data.timestamp = new Date().toISOString();

      // --- NEW GEOLOCATION LOGIC (Vercel Method) ---
      // Read the country and city from Vercel's injected headers
      data.country = req.headers['x-vercel-ip-country'] || 'Unknown';
      data.city = req.headers['x-vercel-ip-city'] || 'Unknown';
      // You can also get the state/region if you want
      // data.region = req.headers['x-vercel-ip-country-region'] || 'Unknown';
      // ---------------------------------------------

      // Insert data into Supabase
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

  res.setHeader('Allow', ['POST', 'OPTIONS']);
  res.status(405).json({ error: `Method ${req.method} Not Allowed` });
};