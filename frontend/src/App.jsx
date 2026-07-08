import { useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [message, setMessage] = useState("");
  const [recommendations, setRecommendations] = useState([]);

  async function handleUpload() {
    if (!selectedFile) {
      setMessage("Please choose a CSV first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("http://127.0.0.1:8000/recommend", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    setMessage(data.message);
    setRecommendations(data.recommendations || []);
  }

  return (
    <div>
      <h1>Movie Recommender</h1>

      <input
        type="file"
        accept=".csv"
        onChange={(e) => setSelectedFile(e.target.files[0])}
      />

      {selectedFile && <p>Selected: {selectedFile.name}</p>}

      <button onClick={handleUpload}>Get Recommendations</button>

      <p>{message}</p>

      <ul>
        {recommendations.map((movie, index) => (
          <li key={index}>
            {movie.title} — {movie.score}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;