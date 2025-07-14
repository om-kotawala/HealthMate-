<?php
include_once 'config.php';

$query = "SELECT * FROM appointments";
$stmt = $conn->prepare($query);
$stmt->execute();

$appointments = $stmt->fetchAll(PDO::FETCH_ASSOC);
?>

<!DOCTYPE html>
<html>
<head>
    <title>Admin - Manage Appointments</title>
</head>
<body>
    <h1>Manage Appointments</h1>
    <table border="1">
        <tr>
            <th>ID</th>
            <th>Patient Name</th>
            <th>Doctor Name</th>
            <th>Appointment Time</th>
            <th>Status</th>
            <th>Actions</th>
        </tr>
        <?php foreach($appointments as $appointment): ?>
        <tr>
            <td><?php echo $appointment['id']; ?></td>
            <td><?php echo $appointment['patient_name']; ?></td>
            <td><?php echo $appointment['doctor_name']; ?></td>
            <td><?php echo $appointment['appointment_time']; ?></td>
            <td><?php echo $appointment['status']; ?></td>
            <td>
                <a href="edit_appointment.php?id=<?php echo $appointment['id']; ?>">Edit</a>
                <a href="delete_appointment.php?id=<?php echo $appointment['id']; ?>">Delete</a>
            </td>
        </tr>
        <?php endforeach; ?>
    </table>
</body>
</html>
?>
