<?php
session_start();
require_once "config.php";

if (!isset($_SESSION['admin'])) {
    header("Location: admin_login.php");
    exit();
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = $_POST['name'];
    $specialization = $_POST['specialization'];
    $fees = $_POST['fees'];
    $contact = $_POST['contact'];

    $stmt = $conn->prepare("INSERT INTO doctors (name, specialization, fees, contact) VALUES (?, ?, ?, ?)");
    $stmt->execute([$name, $specialization, $fees, $contact]);
    $message = "Doctor added successfully.";
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add Doctor</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background: #f4f4f4; }
        .sidebar { width: 250px; height: 100vh; background: #333; color: white; position: fixed; top: 0; left: 0; padding: 20px; }
        .sidebar h2 { text-align: center; margin-bottom: 30px; }
        .sidebar ul { list-style: none; padding: 0; }
        .sidebar ul li { margin: 20px 0; }
        .sidebar ul li a { color: white; text-decoration: none; font-size: 18px; display: block; }
        .sidebar ul li a:hover { background: #575757; padding-left: 10px; transition: 0.3s; }
        .main-content { margin-left: 270px; padding: 20px; }
        .container { width: 50%; margin: 50px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        h1 { color: #333; }
        .form-group { margin: 20px 0; }
        .form-group label { display: block; margin-bottom: 5px; }
        .form-group input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .btn { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; transition: background 0.3s; }
        .btn:hover { background: #0056b3; }
        .message { color: green; }
        .doctor-list { margin-top: 20px; }
        .doctor-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px; }
        .doctor-item button { background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; }
        .doctor-item button:hover { background: #c82333; }
    </style>
</head>
<body>

<div class="sidebar">
    <h2>Admin Panel</h2>
    <ul>
        <li><a href="dashboard.php">Dashboard</a></li>
        <li><a href="add_doctor.php">Add Doctor</a></li>
        <li><a href="view_doctors.php">View Doctors</a></li>
        <li><a href="logout.php">Logout</a></li>
    </ul>
</div>

<div class="main-content">
    <div class="container">
        <h1>Add Doctor</h1>
        <?php if (isset($message)) : ?>
            <p class="message"><?php echo $message; ?></p>
        <?php endif; ?>
        <form method="POST">
            <div class="form-group">
                <label for="name">Name</label>
                <input type="text" id="name" name="name" required>
            </div>
            <div class="form-group">
                <label for="specialization">Specialization</label>
                <input type="text" id="specialization" name="specialization" required>
            </div>
            <div class="form-group">
                <label for="fees">Fees</label>
                <input type="number" id="fees" name="fees" required>
            </div>
            <div class="form-group">
                <label for="contact">Contact</label>
                <input type="text" id="contact" name="contact" required>
            </div>
            <button type="submit" class="btn">Add Doctor</button>
        </form>
        <div class="doctor-list">
            <h2>Doctors List</h2>
            <?php
            $stmt = $conn->prepare("SELECT * FROM doctors");
            $stmt->execute();
            $doctors = $stmt->fetchAll();
            foreach ($doctors as $doctor) {
                echo '<div class="doctor-item">';
                echo '<span>' . $doctor['name'] . ' - ' . $doctor['specialization'] . ' - ' . $doctor['fees'] . ' - ' . $doctor['contact'] . '</span>';
                echo '<button onclick="deleteDoctor(' . $doctor['id'] . ')">Delete</button>';
                echo '</div>';
            }
            ?>
        </div>
    </div>
</div>

<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
<script>
    function deleteDoctor(id) {
        if (confirm('Are you sure you want to delete this doctor?')) {
            window.location.href = 'delete_doctor.php?id=' + id;
        }
    }
</script>
</body>
</html>
