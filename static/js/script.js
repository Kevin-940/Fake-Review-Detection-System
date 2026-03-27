let chart;

function toggleUploadMenu() {
  const menu = document.getElementById("uploadMenu");
  if (menu.style.display === "block") {
    menu.style.display = "none";
  } else {
    menu.style.display = "block";
  }
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload_csv", {
    method: "POST",
    body: formData
  })
  .then(response => response.text())
  .then(html => {
    document.open();
    document.write(html);
    document.close();
  })
  .catch(error => {
    alert("Error uploading CSV");
    console.error(error);
  });
}

function uploadImage(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("image", file);

  fetch("/upload_image", {
    method: "POST",
    body: formData
  })
  .then(res => res.text())
  .then(html => {
    document.open();
    document.write(html);
    document.close();
  });
}

async function analyzeReviews() {
  const text = document.getElementById("reviewText").value.trim();
  if (!text) {
    alert("Please enter a review");
    return;
  }

  const reviews = text.split("\n").filter(r => r.trim());

  // Single review → results page
  if (reviews.length === 1) {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: "review=" + encodeURIComponent(reviews[0])
    });

    const data = await response.json();

    if (data.analysis_id) {
      window.location.href = "/results/" + data.analysis_id;
    } else {
      alert("Error analyzing review");
    }
    return;
  }

  // Multiple reviews → show chart
  let fakeCount = 0;
  let genuineCount = 0;
  const reviewList = document.getElementById("reviewList");
  reviewList.innerHTML = "";

  for (let review of reviews) {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body: "review=" + encodeURIComponent(review)
    });

    const data = await response.json();

    const div = document.createElement("div");
    div.classList.add("review-item");

    if (data.result === "Fake") {
      div.classList.add("fake");
      fakeCount++;
    } else {
      div.classList.add("genuine");
      genuineCount++;
    }

    div.innerHTML = `<strong>${data.result}</strong> - ${review}`;
    reviewList.appendChild(div);
  }

  document.getElementById("totalCount").innerText = reviews.length;
  document.getElementById("fakeCount").innerText = fakeCount;
  document.getElementById("genuineCount").innerText = genuineCount;

  document.getElementById("inputView").classList.add("hidden");
  document.getElementById("resultsView").classList.remove("hidden");

  const ctx = document.getElementById("pieChart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Fake", "Genuine"],
      datasets: [{
        data: [fakeCount, genuineCount],
        backgroundColor: ["#ef4444", "#22c55e"]
      }]
    }
  });
}

function goBack() {
  document.getElementById("resultsView").classList.add("hidden");
  document.getElementById("inputView").classList.remove("hidden");
}