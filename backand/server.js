const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();

app.use(cors());
app.use(bodyParser.json());

// Dummy login (no database)
app.post("/login", (req, res) => {
  const { username, password } = req.body;

  // Simple hardcoded login
  if (username === "admin" && password === "1234") {
    res.json({ success: true });
  } else {
    res.json({ success: false });
  }
});

app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
