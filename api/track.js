const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL; // From Vercel env vars
const supabaseKey = process.env.SUPABASE_ANON_KEY; // From Vercel env vars
const supabase = createClient(supabaseUrl, supabaseKey);

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const data = req.body; // Data sent from the JS tracking script
  data.ip_address = req.headers['x-forwarded-for']?.split(',') || req.connection.remoteAddress; // Capture IP server-side
  data.timestamp = new Date().toISOString(); // Auto-add timestamp if not provided

  // Optional: Enrich country/city from IP (uncomment and install 'axios' if desired)
  // const response = await require('axios').get(`https://ipapi.co/${data.ip_address}/json/`);
  // data.country = response.data.country_name;
  // data.city = response.data.city;

  const { error } = await supabase.from('analytics').insert([data]); // Insert into your Supabase table

  if (error) {
    console.error(error);
    return res.status(500).json({ error: error.message });
  }

  res.status(200).json({ success: true });
};